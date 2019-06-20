"""
Author: poiroot
"""

import logging
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .metrics import Metric, getModelCoefs
from ..utils.load import load_data, load_model

logger = logging.getLogger(__name__)

class FortuneTeller:
    """
    The predict system

    Parameters
    ----------
    X : pandas DataFrame
        The forecast data

    name : string
        The config target name

    length : int
        The row number of history data

    mean : pandas DataFrame, shape [X.shape[1]]
        The mean of history data, i.e. the data seen by models

    std : pandas DataFrame, shape [X.shape[1]]
        The standard deviation of history data

    cov : pandas DataFrame, shape [X.shape[1], X.shape[1]]
        The covariance matrix of hostory data

    models :
        The models created by sklearn

    Attributes
    ----------
    r2 : float
        The maximum R^2 of models

    RI : float
        The confidence of predict value by system

    MD : float
        The mahalannobis distance of X

    DQIX : float
        The data quality of X computed by multivariate normal distribution

    ZScore : pandas DataFrame, shape [X.shape[1]]
        The z-score of X

    predicts : dict, shape [length of models], key : model, value : float
        The predictive values by models

    predict :float
        The predictive value by weighted average of models

    coefs : dict, shape [length of models], key : model, value : float
        The coefficients of each algorithms

    intercept : float
        The intercept of linear regression model

    error : float
        The mean absolute error of the linear regression model

    algo_name : list
        The algorithm name in config "algo_name"

    mae : float
        The mean absolute error of models

    threshold: float, optional, default 0
        Get predict values whose confidence index greater than threshold

    time : datetime object
        The predict time i.e. get feature time
    """

    def __init__(self, name, data_manager, sql):

        self.time = None
        self.X = None
        self.data_manager = data_manager
        self.config = data_manager.model_resource[name]
        self.name = name
        self.sql = sql
        self.length = self.config["train"].data.shape[0]
        self.mean = self.config["mean"]
        self.std = self.config["std"]
        self.cov = self.config["cov"]
        self.models = self.config["models"]
        self.feature_shapes = self.config["feature_shapes"]
        self.r2 = max(self.config["algo_r2"])
        self.RI = 0
        self.MD = 0
        self.DQIX = 0
        self.ZScore = 0
        self.predicts = dict()
        self.predict = 0
        self.coefs = dict()
        self.intercept = 0
        self.error = 0
        self.algo_name = [name.split(".")[0] for name in self.config['algo_name']]
        for name in self.algo_name:
            self.coefs[name] = 1 / len(self.models)
        self.mae = 0
        self.threshold = self.config["threshold"]
        self.start_time = None
        self.target_data = None
        self.metric = Metric(self.config["train"].data, self.r2, self.config["confidence"])
        self.contribute = list()
        self.error_method = "mse"
        logger.info("Creating as instance of FortuneTeller")

    def tell(self, time):
        """
        The main method of system
        """
        X = self.data_manager.get_feature(self.name, time)
        self.time = time
        self.X = X
        self.MD, self.contribute = self.metric.mahalanobis(X=X, mean=self.mean, cov=self.cov, length=self.length)
        if self.config["confidence"] is True:
            self.DQIX = self.metric.quality(X=X, mean=self.mean, cov=self.cov, log=True)
            self.RI = self.metric.ri(forecast_DQIX=self.DQIX)
        logger.info("Compute indicator score")

        for name, model, feature_shape in zip(self.algo_name, self.models, self.feature_shapes):
            if feature_shape is None:
                predict = model.predict(X)
            else:
                shape = [-1] + [i for i in feature_shape[1:]]
                X = np.array(X).reshape(shape)
                predict = model.predict(X)
            while True:
                if len(predict.shape) > 1:
                    predict = predict[0]
                else:
                    break
            self.predicts[name] = float(predict[0])
        logger.info("Predict")

        end_time = time
        logger.debug("target_data time: {0}~{1}".format(self.start_time, end_time))
        self.target_data = self.data_manager.get_target(self.name, self.start_time, end_time, ["SAMPLED_DATE", "RESULT_VALUE"])
        logger.info("Get target data")
        logger.debug("target_data length: {0}".format(self.target_data.shape[0]))

        self.coefs = self._updateModelCoefs(self.coefs)
        if self.config['revise'] is True:
            self._revise()

        if self.target_data.shape[0] >= self.config['revise_sample_times']:
            self.start_time = self.target_data.iloc[1]['time']

        self.predict = 0
        for name in self.algo_name:
            self.predict += self.coefs[name] * self.predicts[name]
        self.predict += self.intercept

        self.mae = self._calculateMAE()

    def _updateModelCoefs(self, weights):
        if self.target_data.shape[0] < self.config["revise_sample_times"]:
            return weights
        error = dict()
        value = defaultdict(list)
        for name in self.algo_name:
            self.drop_index = list()
            for index in self.target_data.index:
                row = self.target_data.loc[index]
                predict_value = self._getPredictValue(row["time"],
                                                      name,
                                                      self.config["revise_minutes_low"],
                                                      self.config["revise_minutes_high"])
                if np.isinf(predict_value):
                    self.drop_index.append(index)
                    continue
                value[name].append(predict_value)
            true_value = self.target_data["value"].copy()
            true_value.drop(self.drop_index, inplace=True)
            error[name] = self._calculateError(true_value,
                                               value[name],
                                               method=self.error_method)
        if len(weights) == 1:
            return weights
        if sum(error.values()) == 0:
            return weights
        new_weights = getModelCoefs(error)
        return new_weights

    def _revise(self):
        """
        The method that revise the predict value by simple math

        Should check quality of y ??
        According LIMS time to get predict value then revise
        Because predict value is immediate

        """

        if self.target_data.shape[0] < self.config["revise_sample_times"]:
            return

        value = list()
        for index in self.target_data.index:
            row = self.target_data.loc[index]
            value.append(self._getPredictValue(row["time"],
                                               self.algo_name,
                                               self.config["revise_minutes_low"],
                                               self.config["revise_minutes_high"]))
        predict = list()
        for part in value:
            num = 0
            for i, algo in enumerate(self.algo_name):
                if not np.isinf(part[i]):
                    num += self.coefs[algo] * part[i]
            if num != 0:
                predict.append(num)
        true_value = self.target_data["value"].copy()
        true_value.drop(self.drop_index, inplace=True)
        self.error = self._calculateError(true_value, predict, method="mae")

        try:
            self.intercept = sum(true_value - predict) / len(predict)
        except ZeroDivisionError:
            self.intercept = 0
        logger.info("Revise")

    def _getPredictValue(self, lims_time, name=None, start_time=-30, end_time=0):
        """
        According the LIMS time and the time interval to get the mean of predictive value
        Should delete extreme predictive values??

        Parameters
        ----------
        lims_time : datetime
            The LIMS time

        name : string or list
            The algorithm name

        start_time : int, optional, default -30
            The start time of the time interval according LIMS time
            i.e. start time of the time interval = LIMS time + start_time

        end_time : int, optional, default 0
            The end time of the time interval according LIMS time
            i.e. end time of the time interval = LIMS time + end_time

        Returns
        -------
        value : float or list
            The mean of predictive values in the time interval
        """
        start = lims_time + timedelta(minutes=start_time)
        end = lims_time + timedelta(minutes=end_time)

        # get RTPMS data and mean it
        predict_df = self.sql.get_model_result(self.name, start, end)
        try:
            predict_df = predict_df[predict_df["conf_idx"] > self.threshold]
            if predict_df.shape[0] == 0:
                logger.debug("There is no {0} data's confidence index greater than threshold, Time: {1} ~ {2}".format(name, start, end))
                if type(name) is list:
                    return [np.inf for i in name]
                return np.inf
            if name is None:
                value = np.mean(predict_df["predict"])
            elif type(name) is list:
                value = list()
                for n in name:
                    value.append(np.mean(predict_df["predict_{0}".format(n)]))
            else:
                value = np.mean(predict_df["predict_{0}".format(name)])
        except KeyError:
            # There is no data in the database
            logger.error("No {0} data in database, Time: {1} ~ {2}".format(name, start, end))
            if type(name) is list:
                return [np.inf for i in name]
            return np.inf
        return value

    def _calculateError(self, true, predict, method="mse"):
        """
        Calculate error between true values and predictive values

        Parameters
        ----------
        true : list
            The true values

        predict : list
            The predictive values

        method : string, optional, default "mse"
            The method to calculate error
                "mse" : mean square error
                "mae" : mean absolute error
                "r2" : 1 - (R squared)
        """
        methods = set(["mse", "mae", "r2"])
        if method not in methods:
            raise ValueError("The method is invalid")
        if len(true) == 0 or len(predict) == 0:
            return 0

        if method == "mse":
            error = mean_squared_error(true, predict)
        elif method == "mae":
            error = mean_absolute_error(true, predict)
        elif method == "r2":
            error = 1 - r2_score(true, predict)

        return error

    def setStartTime(self, time):
        self.start_time = time

    def _calculateMAE(self):
        """
        Calculating MAE bases on the linear regression model and model difference
        """
        mae = self.error
        model_diff_mae = list()

        for indices in combinations(range(len(self.algo_name)), 2):
            diff = abs(self.predicts[self.algo_name[indices[0]]] - self.predicts[self.algo_name[indices[1]]])
            coefs = [abs(self.coefs[self.algo_name[indices[i]]]) for i in range(2)]
            ratio = min(coefs) / max(coefs)
            if ratio is np.nan:
                continue
            model_diff_mae.append(ratio * diff)
        try:
            mae += max(model_diff_mae)
        except ValueError:
            pass

        return mae

    def initialModelCoefs(self):
        time = datetime.now()
        logger.info("Intialize model coefs start")
        if self.config["target_source"] == "rtpms":
            self.target_data = self.data_manager.get_target(self.name, time - timedelta(seconds=5*self.config["predict_sleep_seconds"]), time, ["SAMPLED_DATE", "RESULT_VALUE"])
        else:
            self.target_data = self.data_manager.get_target(self.name, time - timedelta(days=1), time, ["SAMPLED_DATE", "RESULT_VALUE"])
        error = dict()
        value = defaultdict(list)
        drop_index = list()
        for index in self.target_data.index:
            row = self.target_data.loc[index]
            base_time = row["time"]
            times = [base_time + timedelta(minutes=i) for i in range(self.config["revise_minutes_low"],
                                                                     self.config["revise_minutes_high"],
                                                                     self.config["predict_sleep_seconds"] // 60)]
            predicts = defaultdict(list)
            for t in times:
                if t > time:
                    break
                feature = self.data_manager.get_feature(self.name, t)
                for name, model, feature_shape in zip(self.algo_name, self.models, self.feature_shapes):
                    if feature_shape is None:
                        predict = model.predict(feature)
                    else:
                        shape = [-1] + [i for i in feature_shape[1:]]
                        reshape_feature = np.array(feature).reshape(shape)
                        predict = model.predict(reshape_feature)
                    while True:
                        if len(predict.shape) > 1:
                            predict = predict[0]
                        else:
                            break
                    if self.config["confidence"] is True:
                        self.DQIX = self.metric.quality(X=feature, mean=self.mean, cov=self.cov, log=True)
                        self.RI = self.metric.ri(forecast_DQIX=self.DQIX)
                        if self.RI > self.threshold:
                            predicts[name].append(predict[0])
                    else:
                        predicts[name].append(predict[0])
            if len(predicts.keys()) == 0:
                drop_index.append(index)
            for name in predicts.keys():
                if len(predicts[name]) == 0:
                    drop_index.append(index)
                    break
                value[name].append(np.mean(predicts[name]))
        true_value = self.target_data["value"].copy()
        true_value.drop(drop_index, inplace=True)
        for name in self.algo_name:
            error[name] = self._calculateError(true_value,
                                               value[name],
                                               method=self.error_method)
        weights = dict()
        total_error = sum(error.values())
        if total_error == 0:
            logger.info("Initialize model coefs and intercept done by total error is zero")
            return self.coefs, 0
        if len(self.algo_name) == 1:
            name = self.algo_name[0]
            weights[name] = 1
        else:
            weights = getModelCoefs(error)
        logger.info("Initialize model coefs done")

        if true_value.shape[0] < self.config["revise_sample_times"]:
            return weights, 0

        predict = list()
        for i in range(len(true_value)):
            num = 0
            for key in value:
                num += weights[key] * value[key][i]
            predict.append(num)

        if self.config["revise"]:
            try:
                intercept = sum(true_value - predict) / len(predict)
            except ZeroDivisionError:
                intercept = 0
        else:
            intercept = 0
        logger.info("Initialize intercept done")
        return weights, intercept


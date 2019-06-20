"""
Author: poiroot
"""

import math
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import logging
from scipy import optimize
from numpy.linalg import LinAlgError
from ..utils.validation import load_data

logger = logging.getLogger(__name__)


class Metric():
    """
    Some indicators for system: Mahalanobis Distance, Z_score, quality

    Parameters
    ----------
    X : ndarray
        The forecast data

    history : ndarray
        The historical data i.e. the training data

    r2: float
        The maximum R^2 of training models

    length : int
        The row number of history data

    mean : ndarray, shape [X.shape[1]]
        The mean of history data, i.e. the data seen by models

    std : ndarray, shape [X.shape[1]]
        The standard deviation of history data

    cov : ndarray, shape [X.shape[1], X.shape[1]]
        The covariance matrix of hostory data

    Attributes
    ----------
    MD : list
        The MD distribution of historical data

    """
    def __init__(self, history, r2, confidence):
        if confidence is True:
            logger.info("Calculate history DQIX")
            self.DQIX = self._calculateHistoryDQIX(history)
            self.revise_std = self._calculateReviseStd(r2)
            logger.debug("History DQIX mean: {0}, std: {1}".format(np.mean(self.DQIX), np.std(self.DQIX)))
            logger.debug("Revise std: {0}".format(self.revise_std))

    def mahalanobis(self, X, mean, cov, length, bias=False, pool=True):
        """
        This indicator implies the new data is in the historical data distribution.
        The smaller distance means the new data has more opportunity in the historical data distribution.

        Parameters
        ----------
        bias : boolean, optional, default False
            Default normalization (False) is by (N - 1), where N is the number of observations given (unbiased estimate).
            If bias is True, then normalization is by N.

        pool : boolean, optional, default True
            Default means the covariance matrix of two groups is weighted sum of each covariance matrix.
            If pool is False, use average of each covariance matrix.

        Outputs
        -------
        distance : float
            The mahalanobis distance

        contribute : ndarray
            The contribute of each tags

        """
        mean_X = np.mean(X, axis=0)
        X = X - mean_X
        cov_X = np.cov(X, rowvar=False, bias=bias)

        cov_mix = (cov + cov_X) / 2
        if pool:
            total_row = length + X.shape[0]
            cov_mix = (length / total_row) * cov + (X.shape[0] / total_row) * cov_X

        try:
            inv_cov = np.linalg.inv(cov_mix)
        except LinAlgError:
            inv_cov = np.linalg.pinv(cov_mix)

        diff_mean = mean - mean_X
        part_equation = diff_mean.T.dot(inv_cov)
        distance = 0
        contribute = dict()
        for i in range(len(diff_mean)):
            d = part_equation[i] * diff_mean[i]
            contribute[X.columns[i]] = [d]
            distance += d
        contribute = pd.DataFrame.from_dict(contribute)
        distance = math.sqrt(distance)
        # distance = math.sqrt((diff_mean.T.dot(inv_cov)).dot(diff_mean))

        return distance, contribute

    def z_score(self, X, mean, std):
        """
        This indicator is Z score.
        According history data mean and standrad deviation to calculate X's Z score

        Return the higher z-score feature ??

        Outputs
        -------
        M : ndarray (2-D)
            The Z score of X according history data mean and standrad deviation
        """
        M = (X - mean) / std
        return M

    def quality(self, X, mean, cov, log=False):
        """
        Consider dependent varaible ??
        This indicator implies the probability of X in history data.
        Use multivariate normal distribution

        Parameters
        ----------
        log : boolean, optional, default False
            If True, calculate log of probability.
            If False, calculalte probability.

        Outputs
        -------
        p : float or ndarray [rows of group]
            The probability
        """
        if log:
            p = scipy.stats.multivariate_normal.logpdf(X, mean=mean, cov=cov, allow_singular=True)
        else:
            p = scipy.stats.multivariate_normal.pdf(X, mean=mean, cov=cov, allow_singular=True)

        return p

    def ri(self, forecast_DQIX):
        """
        This indicator implies the confidence of predict value

        Parameters
        ----------
        forecast_DQIX : float
            The data quality of forcast data

        Outputs
        -------
            The confidence value between 0 and 100
        """

        std_is_too_small = False

        if std_is_too_small:
            confidence = (1 - abs(forecast_DQIX - np.mean(self.DQIX)) / abs(np.mean(self.DQIX))) * 100
            if confidence < 0:
                return 0
            return confidence

        p = scipy.stats.norm.cdf(forecast_DQIX, np.mean(self.DQIX), self.revise_std)

        return 1 - abs(0.5 - p) * 2

    def _calculateHistoryDQIX(self, X, bias=False):

        DQIX = list()
        latest_percent = 0

        for i, index in enumerate(X.index):
            part_X = X.drop(index)
            mean = np.mean(part_X)
            cov = np.cov(part_X, rowvar=False, bias=bias)
            DQIX.append(self.quality(X.loc[index], mean, cov, True))
            percent = int(i / len(X.index) * 100)
            if latest_percent != percent:
                latest_percent = percent
                if latest_percent % 5 == 0:
                    logger.debug("{0}% Done".format(latest_percent))

        return DQIX

    def _calculateReviseStd(self, r2):

        mean = np.mean(self.DQIX)
        std = np.std(self.DQIX)
        target = mean - 2 * std
        prob = 0.95 * r2 / 2

        def objective(x, *param):

            return abs(prob - scipy.stats.norm.cdf(target, loc=mean, scale=x*std))

        ranges = (slice(1, 100, 0.1),)
        resbrute = optimize.brute(objective, ranges, full_output=True, finish=optimize.fmin)

        return resbrute[0][0] * std


def getModelCoefs(error):
    """
    Calculate new weights based on error, the larger error, the smaller weight

    Parameters
    ----------
    error: dict
        The error of each models

    Returns
    -------
    new_weights: dict
        The coefs of each models
    """
    new_weights = dict()
    total_weight = 1
    while len(error) > 1:
        weight = dict()
        for key in error.keys():
            weight[key] = 1 - error[key] / sum(error.values())
        max_weight_keys = [k for k, v in weight.items() if v == max(weight.values())]
        for key in max_weight_keys:
            new_weights[key] = total_weight * weight[key] / len(max_weight_keys)
            error.pop(key)
        total_weight -= new_weights[max_weight_keys[0]] * len(max_weight_keys)
    for key in error.keys():
        new_weights[key] = total_weight
    return new_weights




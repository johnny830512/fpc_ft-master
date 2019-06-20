"""
Author: poiroot
"""

import oyaml as yaml
import numpy as np
import pandas as pd
from keras.models import load_model as lm
from datetime import datetime as dt
from sklearn.externals import joblib
from pycode.feature.core.reader import FeatureReader


def load_data(filepath=None, target_columns=-1):
    """
    Load history data
    For now, it only support csv file

    Parameters
    ----------
    filepath : string, default None
        The data file path

    target_columns : int or string, optional, default -1
        The target columns in data file

    Returns
    -------
    Data class: have data, target attributes
    """
    class Data:
        """
        The history data class
        """

        def __init__(self):
            self.data = None
            self.target = None

    if filepath is None:
        raise ValueError("The filepath is None, please check the filepath is in the config file")

    df = pd.read_csv(filepath)
    data = Data()
    columns = list(df.columns)
    if type(target_columns) is int:
        columns.pop(target_columns)
        data.target = df[df.columns[target_columns]]
    if type(target_columns) is str:
        columns.remove(target_columns)
        data.target = df[target_columns]
    data.data = df[columns]

    return data


def load_model(filepath=None):
    """
    Load pre-trained model
    Support the model that created by sklearn.externals.joblib

    Parameters
    ----------
    filepath : string, default None
        The model file name

    Returns
    -------
    The model created by sklearn, xgboost or keras, and the model input shape
    """

    if filepath is None:
        raise ValueError("The filepath is None, please check the filepath is in the config file")
    if '.h5' in filepath:
        keras_model = lm(filepath)
        feature_shape = keras_model.layers[0].input_shape
        array_shape = [1] + [i for i in feature_shape[1:]]
        array = np.zeros(array_shape)
        # for keras bug
        keras_model.predict(array)
        return keras_model, feature_shape
    else:
        return joblib.load(filepath), None


def load_config(filepath=None):
    """
    Load config .yaml file

    Parameters
    ----------
    filepath : string, default None
        The model file name

    Returns
    -------
    The dictionary
    """
    if filepath is None:
        raise ValueError("The filepath is None, please check the config file is exist")

    with open(filepath, "r") as stream:
        output = dict()
        try:
            content = yaml.load(stream)
            output.update(content)
            return output
        except yaml.YAMLError as e:
            print(e)

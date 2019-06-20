"""
Author: poiroot
"""
from .load import load_data, load_model
import numpy as np


def check_resources(config):
    """
    Check the resources for FortuneTeller class

    Parameters
    ----------
    config : dict
        The dictionary read from yaml config file

    Returns
    -------
    output : dict
        The resources for FortuneTeller class
    """
    output = config

    for element in config:
        if element == 'predict_items':
            for item in output[element].keys():
                prep_steps = list()
                for name in config[element][item]["prep_name"]:
                    scaler, _ = load_model(config[element][item]["prep_dir_path"] + name)
                    prep_steps.append(scaler)
                output[element][item]["prep_steps"] = prep_steps

                output[element][item]["train"] = load_data(config[element][item]["data_dir_path"] + config[element][item]["data_name"], config[element][item]["data_target"])
                output[element][item]["models"] = list()
                output[element][item]["feature_shapes"] = list()
                model_filepaths = list()
                for name in config[element][item]["algo_name"]:
                    model_filepaths.append(config[element][item]["algo_dir_path"] + name)
                for model_filepath in model_filepaths:
                    model, feature_shape = load_model(model_filepath)
                    output[element][item]["models"].append(model)
                    output[element][item]["feature_shapes"].append(feature_shape)
                output[element][item]["mean"] = np.mean(output[element][item]["train"].data, axis=0)
                output[element][item]["std"] = np.std(output[element][item]["train"].data, axis=0, ddof=0)
                output[element][item]["cov"] = np.cov(output[element][item]["train"].data, rowvar=0)

    return output

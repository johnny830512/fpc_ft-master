"""
Author: poiroot
"""

import yaml
import copy


def write_config(config, filepath):
    """
    remove input config
        train
        models
        mean
        std
        cov
    then write to config file

    Parameters
    ----------
    config : dict
        The dict read from yaml file
    """
    remove_keys = ["train", "models", "mean", "std", "cov"]
    output = copy.deepcopy(config)
    for element in output:
        if element == "sql_connect":
            continue
        for item in output[element].keys():
            for key in remove_keys:
                del output[element][item][key]
    temp = copy.deepcopy(output)
    output = list()
    for key in temp:
        D = dict()
        D[key] = temp[key]
        output.append(D)
    with open(filepath, "w") as outfile:
        yaml.dump(output, outfile, default_flow_style=False)

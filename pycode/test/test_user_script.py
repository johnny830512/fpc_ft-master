import unittest
import numpy as np
import pandas as pd
import inspect
import re
import os
import importlib
import env
from pycode.utils.load import load_config, load_data
from pycode.utils.validation import check_resources
from datetime import datetime as dt

class FeatureTransformTest(unittest.TestCase):

    user_script = None

    def setUp(self):
        self.config = load_config("config.yaml")
        self.config = check_resources(self.config)

        # Dynamic import
        pysearchre = re.compile(".py$", re.IGNORECASE)
        transform_files = filter(pysearchre.search,
                                 os.listdir(os.path.join(os.path.dirname(__file__),
                                                         "../feature/userscr")))
        form_module = lambda fp: "." + os.path.splitext(fp)[0]
        transforms = map(form_module, transform_files)
        importlib.import_module("pycode.feature.userscr")
        modules = []
        for transform in transforms:
            if not transform.startswith(".__") and transform.startswith("." + self.user_script):
                modules.append(importlib.import_module(transform, package="pycode.feature.userscr"))

        self.class_members = [m for m in inspect.getmembers(modules[0], inspect.isclass)
                              if m[1].__module__ == "pycode.feature.userscr.{0}".format(self.user_script)]
        self.predict_items = list()
        for member in self.class_members:
            obj = member[1](self.config)
            self.predict_items.append(obj.predict_name)

        self.columns = dict()
        for item in self.predict_items:
            data_filepath = self.config["predict_items"][item]["data_dir_path"] + self.config["predict_items"][item]["data_name"]
            data = load_data(data_filepath, self.config['predict_items'][item]['data_target'])
            self.columns[item] = list(data.data.columns)

    def tearDown(self):
        self.config = None

    def testConfigAttributes(self):
        true_label = set(["device",
                          "note",
                          "data_name",
                          "data_target",
                          "algo_name",
                          "algo_r2",
                          "algo_dir_path",
                          "data_dir_path",
                          "prep_dir_path",
                          "prep_name",
                          "confidence",
                          "confidence_bounds",
                          "predict_sleep_seconds",
                          "revise",
                          "revise_minutes_low",
                          "revise_minutes_high",
                          "revise_sample_times",
                          "threshold",
                          "target_source",
                          "sample_point",
                          "sample_item",
                          "grade_list",
                          "tags",
                          "train",
                          "mean",
                          "std",
                          "cov",
                          "models",
                          "prep_steps",
                          "feature_shapes"])
        for item in self.predict_items:
            with self.subTest(predict_item=item):
                self.assertEqual(true_label, set(self.config["predict_items"][item]), msg="\n{0} Config attributes Error".format(item))

    def testConfigAttributeTypes(self):
        true_label = {"device": str,
                      "note": str,
                      "data_name" : str,
                      "data_target" : (str, type(None)),
                      "algo_name" : list,
                      "algo_r2" : list,
                      "algo_dir_path" : str,
                      "data_dir_path" : str,
                      "prep_dir_path": str,
                      "prep_name": list,
                      "confidence": bool,
                      "confidence_bounds": list,
                      "predict_sleep_seconds" : int,
                      "revise": bool,
                      "revise_minutes_low" : int,
                      "revise_minutes_high" : int,
                      "revise_sample_times" : int,
                      "threshold" : float,
                      "target_source" : str,
                      "sample_point" : (list, type(None)),
                      "sample_item" : (list, type(None)),
                      "grade_list" : (list, type(None)),
                      "tags" : (list, type(None))}
        for item in self.predict_items:
            with self.subTest(predict_item=item):
                for key in true_label:
                    with self.subTest(attribute=key):
                        self.assertIsInstance(self.config["predict_items"][item][key], true_label[key], msg="\n{0} Config {1} attributes type Error".format(item, key))

    def testClassAttributePredictItems(self):
        true_label = set(self.config["predict_items"].keys())
        test_label = set(self.predict_items)
        msg = "\nClass attribute 'predict_name' must match config order (predict_items[i])"
        self.assertEqual(true_label, test_label, msg)

    def testTransformReturnType(self):
        for member, item in zip(self.class_members, self.predict_items):
            with self.subTest(predict_item=item):
                obj = member[1](self.config)
                self.assertIsInstance(obj.transform(dt.now()), pd.DataFrame, msg="\n{0} class transform method error\nReturn type should be pandas DataFrame".format(obj.__class__.__name__))

    def testTransformReturnColumns(self):
        for member, item in zip(self.class_members, self.predict_items):
            with self.subTest(predict_item=item):
                obj = member[1](self.config)
                columns = list(obj.transform(dt.now()).columns)
                self.assertEqual(columns, self.columns[item], msg="\n{0} class transform method Error\nDataFrame columns order is not match train data columns".format(obj.__class__.__name__))

    def testTransformReturnShape(self):
        for member, item in zip(self.class_members, self.predict_items):
            with self.subTest(predict_item=item):
                obj = member[1](self.config)
                df = obj.transform(dt.now())
                self.assertEqual(1, df.shape[0], msg="\n{0} class transform method Error\nDataFrame must be one row".format(obj.__class__.__name__))

    def testTransformReturnValueIsNan(self):
        for member, item in zip(self.class_members, self.predict_items):
            with self.subTest(predict_item=item):
                obj = member[1](self.config)
                df = obj.transform(dt.now())
                self.assertEqual(False, np.isnan(df.values).any(), msg="\n{0} class transform method Error\nDataFrame has Nan value".format(obj.__class__.__name__))

    def testTransformReturnShapeIsMatchModelInput(self):
        for member, item in zip(self.class_members, self.predict_items):
            with self.subTest(predict_item=item):
                obj = member[1](self.config)
                df = obj.transform(dt.now())
                for model, feature_shape in zip(self.config["predict_items"][item]["models"], self.config["predict_items"][item]["feature_shapes"]):
                    with self.subTest(model=model.__class__):
                        if feature_shape is None:
                            value = model.predict(df)
                        else:
                            shape = [-1] + [i for i in feature_shape[1:]]
                            array = np.array(df).reshape(shape)
                            value = model.predict(array)
                        print("Your predict value is {0}".format(value))
        print("If testTransformReturnColumns is pass, your model is incorrect. Please check your model")
        print("If testTransformReturnColumns is failed, please check your feature_script.py")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        FeatureTransformTest.user_script = sys.argv.pop()
    unittest.main(verbosity=2)

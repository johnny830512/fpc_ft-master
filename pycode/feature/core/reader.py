import inspect
from ..userscr import feature_script


class FeatureReader():

    def __init__(self, config):

        self.config = config
        self.clsmembers = [m for m in inspect.getmembers(feature_script, inspect.isclass)
                      if m[1].__module__ == 'pycode.feature.userscr.feature_script']
        self.feautre_objs = dict()
        for member in self.clsmembers:
            obj = member[1](self.config)
            self.feautre_objs[obj.predict_name] = obj

    def get_feature(self, sample_time):

        features = dict()
        for key in self.feautre_objs.keys():
            obj = self.feautre_objs[key]
            features[obj.predict_name] = obj.transform(sample_time)
        return features

    def get_feature_by_predict_item(self, name, sample_time):
        obj = self.feautre_objs[name]
        return obj.transform(sample_time)

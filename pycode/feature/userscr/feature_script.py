import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
from .template import FeatureTransform

class SAPTeaFeatrue(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[0]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        # overwriting

    def _get_data(self, time):
        # overwriting


    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe



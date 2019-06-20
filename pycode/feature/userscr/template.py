"""
Author: poiroot
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from ...rtpms.RTPMS import RTPMS_OleDB
from ...lims.lims import Lims


class FeatureTransform():

    def __init__(self, config):
        self.author = None
        self.config = config
        self.predict_items = list(config['predict_items'].keys())
        rtpms = create_engine(config['sql_connect']['rtpms'])
        self.rtpms = RTPMS_OleDB(rtpms)
        lims_engine = create_engine(config['sql_connect']['lims'])
        lims_server = config['lims_setting']['history_linked_server']
        lims_table = config['lims_setting']['history_view']
        self.lims = Lims(lims_engine, lims_server, lims_table)
        self.scaler_dict = dict()
        self.prep_steps_dict = dict()
        for key in self.predict_items:
            self.prep_steps_dict[key] = config['predict_items'][key]['prep_steps']

    def _get_data(self):
        """
        Get RTPMS data

        Return type should be pd.DataFrame()
        """
        # TODO implement this method
        return pd.DataFrame()

    def transform(self, time):
        """
        Transform feature

        Return type should be pd.DataFrame()
        """
        # TODO implement this method
        return pd.DataFrame()


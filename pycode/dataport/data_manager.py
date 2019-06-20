import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
from .rtpms import RTPMS_OleDB
from .lims import Lims
from sqlalchemy import create_engine
from ..feature.core.reader import FeatureReader


class DataManager:

    def __init__(self, config):

        self.lims = None
        self.rtpms = None

        if config['sql_connect']['rtpms'] is not None:
            rtpms_engine = create_engine(config['sql_connect']['rtpms'])
            self.rtpms = RTPMS_OleDB(rtpms_engine)

        if config['sql_connect']['lims'] is not None:
            self.lims_engine = create_engine(config['sql_connect']['lims'])
            self.lims = Lims(self.lims_engine,
                             config['lims_setting']['history_linked_server'],
                             config['lims_setting']['history_view'])

        self.model_resource = config['predict_items']
        self.feature_reader = FeatureReader(config)

    def get_feature(self, predict_item_name, predict_time):
        return self.feature_reader.get_feature_by_predict_item(predict_item_name, predict_time)

    def get_all_feature(self, predict_time):
        return self.feature_reader.get_feature(predict_time)

    def get_rtpms_value(self, tag_list, start_time, end_time, time_step):

        if self.rtpms is None:
            return None
        return self.rtpms.get_rtpms(tag_list, start_time, end_time, time_step)

    def get_rtpms_single_value(self, tag_list, time):

        if self.rtpms is None:
            return None
        return self.rtpms.get_single_value(tag_list, time)

    def get_target(self, predict_item_name, start_time, end_time, column_list=None):

        if self.model_resource[predict_item_name]['target_source'] == 'lims':
            df = self.lims.get_lims(column_list,
                                    self.model_resource[predict_item_name]['sample_point'],
                                    self.model_resource[predict_item_name]['sample_item'],
                                    self.model_resource[predict_item_name]['grade_list'],
                                    start_time, end_time)
            rename_columns = {"SAMPLED_DATE": "time", "RESULT_VALUE": "value"}
            df = df.rename(columns=rename_columns)
            df["time"] = pd.to_datetime(df["time"]).dt.to_pydatetime()
            return df

        if self.model_resource[predict_item_name]['target_source'] == 'rtpms':
            tag_list = self.model_resource[predict_item_name]['tags']
            time_step = str(timedelta(seconds=self.model_resource[predict_item_name]["predict_sleep_seconds"]))
            df = self.rtpms.get_rtpms(tag_list, start_time, end_time, time_step)
            rename_columns = {"{0}".format(df.columns[-1]): "value"}
            df = df.rename(columns=rename_columns)
            df = df.reset_index()
            df["time"] = pd.to_datetime(df["time"]).dt.to_pydatetime()

            return df

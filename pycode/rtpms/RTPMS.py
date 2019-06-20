__author__ = 'john'
import pandas as pd
import os
import sys
import logging
import datetime as dt

logger = logging.getLogger(__name__)

class RTPMS_OleDB():
    """Collect real tim data from RTPMS OleDB
    """
    def __init__(self, rtpms_engine):
        self.rtpms_engine = rtpms_engine
        self.query = "select * from OPENQUERY([MLRTPMS], \'select [time],[tag],[value] from [piarchive]..[piinterp] " \
                     "where [tag] in ({0}) AND timestep = {1} AND [time] BETWEEN {2} AND {3} \')"
        self.time_format = "%Y-%m-%d %H:%M:%S"

    def get_rtpms(self, tag_list, start_time, end_time, time_step):
        """Get RTPMS data from oledb by sql query

        Parameters
        ----------
        tag_list: {python list}

        start_time: python datetime object for RTPMS query started time

        end_time: python datetime object for RTPMS query end time

        time_step: python string type for sample rate eg. 1 minute = 00:01:00, 1 hour = 01:00:00, 10 second = 00:00:10,

        Returns
        -------
        df: pandas DataFrame, column name:[tag_name], index:[time]
        """
        try:
            start_time_str = start_time.strftime(self.time_format)
            end_time_str = end_time.strftime(self.time_format)
            query = self.query.format(','.join(["\'\'" + tag + "\'\'" for tag in tag_list]), "\'\'" + time_step + "\'\'", "\'\'"
                                    + start_time_str + "\'\'", "\'\'" + end_time_str + "\'\'")
            logger.info('get_rtpms, tag: {0}; start: {1}, end: {2}, time_step:{3}'.format(','.join(tag_list), start_time_str, end_time_str, time_step))

            df = pd.read_sql(query, self.rtpms_engine)
            df.fillna(value=0, inplace=True)

            tag_df_list = []
            for tag in df['tag'].unique():
                tag_df = df[df['tag'] == tag].set_index(keys="time")
                del tag_df['tag']
                tag_df.columns = [tag]
                tag_df_list.append(tag_df)

            trans_df = pd.concat(tag_df_list, axis=1)
            return trans_df
        except Exception as e:
            logger.error('get_rtpms, error: {0}, query: {1}'.format(str(e), query))
            return str(e)

    def get_single_value(self, tag_list, time):
        try:
            time_str = time.strftime(self.time_format)
            query = "select * from OPENQUERY([MLRTPMS], 'select [tag],[time],[value] from [piarchive]..[piinterp] where tag in ({0}) and time = {1}')".format(
                ','.join(["\'\'" + tag + "\'\'" for tag in tag_list]), "\'\'" + time_str + "\'\'")
            logger.info('get_single_value, tag:{0}; time:{1}'.format(','.join(tag_list), time_str))
            df = pd.read_sql(query, self.rtpms_engine)
            return df
        except Exception as e:
            logger.error('get_single_value, error:{1}, query:{0}'.format(str(e), query))
            return str(e)

    def to_row(self, data):
        """Change columns shape to tags shape by averaging the row value

        Parameters
        ---------
        data: pandas DataFrame, column name:[time, tag, value]

        Returns
        -------
        data: pandas DataFrame
        tag1, tag2, tag3...
        avg1, avg2, avg3...
        """
        data = data.groupby(['tag'], as_index=False).mean()
        data.index = [1] * data.shape[0]
        data = pd.pivot_table(data, columns=['tag'], index=data.index, values='value')
        return data

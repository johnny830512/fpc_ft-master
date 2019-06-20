__author__ = 'polkmnbv'

import pandas as pd
import logging
import datetime as dt

logger = logging.getLogger(__name__)

class Lims():
    """Collect lims data from FPC LIMS
    """
    def __init__(self, lims_engine, lims_server, lims_table):
        self.engine = lims_engine
        self.table = lims_table
        self.query_str = "select * from openquery({0}, 'SELECT {1} FROM {2} {3} {4} order by sampled_date')"
        self.lims_server = lims_server
        self.lims_table = lims_table
        self.time_format = "%Y-%m-%d %H:%M:%S"

    def get_lims(self, column_list=None, sample_point_list=None, component_list=None, grade_list=None, start_time=None, end_time=None):
        """Get lims data from lims db by sql query

        Parameters
        ----------
        columns_list: {python list}

        sample_point_list: {python list}

        component_list: {python list} test items

        grade_list: {python list}

        start_time: python datetime object lims query started time default None if None get all data

        end_time: python datetime object lims query end time default None if None get all data

        Returns
        -------
        df: pandas DataFrame
        """

        time_condition = ''
        where_condition = list()
        where_str = ''
        columns = "*"
        start_time_str = ''
        end_time_str = ''
        info = dict()

        if column_list is not None:
            columns = ','.join([c for c in column_list])

        if sample_point_list is not None:
            sp_join_str = ','.join(["''" + sp + "''" for sp in sample_point_list])
            where_condition.append("sampling_point in ({0})".format(sp_join_str))
            info['sp'] = sp_join_str

        if component_list is not None:
            cp_join_str = ','.join(["''" + cp + "''" for cp in component_list])
            where_condition.append("component_name in ({0})".format(cp_join_str))
            info['cp'] = cp_join_str

        if grade_list is not None:
            gd_join_str = ','.join(["''" + gd + "''" for gd in grade_list])
            where_condition.append("gradename in ({0})".format(gd_join_str))
            info['gd'] = gd_join_str

        if isinstance(start_time, dt.datetime) and isinstance(end_time, dt.datetime):
            start_time_str = start_time.strftime(self.time_format)
            end_time_str = end_time.strftime(self.time_format)
            where_condition.append("sampled_date between {0} and {1}".format("''" + start_time_str + "''", "''" + end_time_str + "''"))

        where_condition.append("status = ''A''")
        where_condition.append("test_status = ''A''")
        where_condition.append("result_status = ''A''")

        if len(where_condition) != 0:
            where_str = 'where '

        logger.info('get_lims, {0} start: {1}; end: {2}'.format('; '.join(['{0}: {1}'.format(key, info[key]) for key in info.keys()]), start_time_str, end_time_str))
        query = self.query_str.format(self.lims_server, columns, self.lims_table, where_str + ' and '.join(where_condition), time_condition)
        df = pd.read_sql(query, self.engine)
        return df


import traceback
import datetime as dt
import pandas as pd
import numpy as np
from pycode.rtpms.RTPMS import RTPMS_OleDB
from pycode.utils.load import load_config
from pycode.lims.lims import Lims
from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource
from flask_cors import CORS, cross_origin
from sqlalchemy import *
from sqlalchemy.orm import mapper, create_session

app = Flask(__name__)
CORS(app)
api = Api(app)

time_format = "%Y-%m-%d %H:%M:%S"

config = load_config('config.yaml')

# Create lims engine
lims_engine = create_engine(config['sql_connect']['lims'])
lims_server = config['lims_setting']['history_linked_server']
lims_table = config['lims_setting']['history_view']
lims = Lims(lims_engine, lims_server, lims_table)

# Create rtpms engine
rtpms_engine = create_engine(config['sql_connect']['rtpms'])
rtpms = RTPMS_OleDB(rtpms_engine)

# Set teller metadata to get predict result
engine = create_engine(config['sql_connect']['fpc_ft'])
db_session = create_session(bind=engine, autocommit=False)
metadata = MetaData(bind=engine)

# Set web API request parser
parser = reqparse.RequestParser()
parser.add_argument('start')
parser.add_argument('end')
parser.add_argument('time_step')
parser.add_argument('tag_list')
parser.add_argument('count')
parser.add_argument('colList')
parser.add_argument('unit')


def str_to_date(date_str, time_format):
    return dt.datetime.strptime(date_str, time_format)


def statistics_info(values):
    stat_info = dict()
    round_bit = 2
    stat_info['max'] = round(np.max(values), round_bit)
    stat_info['min'] = round(np.min(values), round_bit)
    stat_info['std'] = round(np.std(values), round_bit)
    stat_info['mean'] = round(np.mean(values), round_bit)
    return stat_info


class ModelAPI(Resource):

    def get(self, name):
        class Model(object):
            pass
        try:
            args = parser.parse_args()
            start_time = str_to_date(args['start'], time_format)
            end_time = str_to_date(args['end'], time_format)
            col_list = args['colList'].split(',')
            model_table = Table('Model_' + name, metadata, autoload=True,
                                autoload_with=engine)
            mapper(Model, model_table)
            query_str = db_session.query(Model).filter(Model.time.between(
                start_time, end_time)).statement
            result = pd.read_sql(query_str, engine)
            result['time'] = result['time'].apply(lambda x: x.strftime(time_format))
            result = result[col_list]
            return result.to_dict(orient='records')
        except Exception as e:
            db_session.rollback()
            abort(404, message=traceback.format_exc())


class ModelStatisticsAPI(Resource):

    def get(self, name):
        class Model(object):
            pass
        try:
            args = parser.parse_args()
            start_time = str_to_date(args['start'], time_format)
            end_time = str_to_date(args['end'], time_format)
            model_table = Table('Model_' + name, metadata, autoload=True,
                                autoload_with=engine)
            mapper(Model, model_table)
            query_str = db_session.query(Model).filter(Model.time.between(
                start_time, end_time)).statement
            result = pd.read_sql(query_str, engine)
            result['time'] = result['time'].apply(lambda x: x.strftime(time_format))
            stats_info = statistics_info(result['predict'].values)
            stats_info['name'] = name
            return stats_info
        except Exception as e:
            db_session.rollback()
            abort(404, message=traceback.format_exc())


class TargetAPI(Resource):

    def get(self, name):
        try:
            args = parser.parse_args()
            start_time = str_to_date(args['start'], time_format)
            end_time = str_to_date(args['end'], time_format)
            params = config["predict_items"][name]
            """
            Target Source from LIMS
            """
            if params["target_source"] == "lims":
                lims_data = lims.get_lims(["SAMPLED_DATE", "RESULT_VALUE"],
                                          params["sample_point"],
                                          params["sample_item"],
                                          params['grade_list'],
                                          start_time, end_time)
                if len(lims_data) == 0:
                    return []
                lims_data.rename(columns={"SAMPLED_DATE": "time", "RESULT_VALUE": "value"}, inplace=True)
                lims_data.drop_duplicates(['time'], keep='first', inplace=True)
                lims_data["time"] = lims_data["time"].apply(lambda x: x.strftime(time_format))
                result = lims_data.to_dict("records")
            """
            Target source from RTPMS
            """
            if params["target_source"] == "rtpms":
                rtpms_data = rtpms.get_rtpms(params["tags"], start_time, end_time, "00:01:00")
                rtpms_data.rename(columns={rtpms_data.columns[-1]: "value"}, inplace=True)
                rtpms_data["time"] = rtpms_data.index
                rtpms_data['timestamp'] = rtpms_data['time'].values.astype(np.int64) // 10 ** 9
                rtpms_data['time'] = rtpms_data['time'].apply(lambda x: x.strftime(time_format))
                result = rtpms_data.to_dict("recoeds")
            return result
        except Exception as e:
            abort(404, message=traceback.format_exc())


class TargetStatisticsAPI(Resource):

    def get(self, name):
        try:
            args = parser.parse_args()
            start_time = str_to_date(args['start'], time_format)
            end_time = str_to_date(args['end'], time_format)
            params = config["predict_items"][name]

            values = None
            """
            Target Source from LIMS
            """
            if params["target_source"] == "lims":
                lims_data = lims.get_lims(["SAMPLED_DATE", "RESULT_VALUE"],
                                          params["sample_point"],
                                          params["sample_item"],
                                          params['grade_list'],
                                          start_time, end_time)
                if len(lims_data) == 0:
                    return []
                lims_data.rename(columns={"SAMPLED_DATE": "time", "RESULT_VALUE": "value"}, inplace=True)
                lims_data.drop_duplicates(['time'], keep='first', inplace=True)
                lims_data["time"] = lims_data["time"].apply(lambda x: x.strftime(time_format))
                values = lims_data['value'].values
            """
            Target source from RTPMS
            """
            if params["target_source"] == "rtpms":
                rtpms_data = rtpms.get_rtpms(params["tags"], start_time, end_time, "00:01:00")
                rtpms_data.rename(columns={rtpms_data.columns[-1]: "value"}, inplace=True)
                rtpms_data["time"] = rtpms_data.index
                rtpms_data['time'] = rtpms_data['time'].apply(lambda x: x.strftime(time_format))
                values = rtpms_data['value'].values

            stats_info = statistics_info(values)
            stats_info['name'] = name
            return stats_info
        except Exception as e:
            abort(404, message=traceback.format_exc())


class TagValuesAPI(Resource):

    def get(self):
        try:
            args = parser.parse_args()
            start_time = str_to_date(args['start'], time_format)
            end_time = str_to_date(args['end'], time_format)
            time_step = args['time_step']
            tag_list = args['tag_list'].split(',')
            rtpms_data = rtpms.get_rtpms(tag_list, start_time, end_time, time_step)
            rtpms_data['time'] = rtpms_data['time'].apply(lambda x: x.strftime(time_format))
            return rtpms_data.to_dict(orient='list')
        except Exception as e:
            abort(404, message=traceback.format_exc())


class ContributeAPI(Resource):

    def get(self, name):
        class Model(object):
            pass
        try:
            args = parser.parse_args()
            start_time = str_to_date(args['start'], time_format)
            end_time = str_to_date(args['end'], time_format)
            model_table = Table('Model_' + name + "_mdist", metadata, autoload=True,
                                autoload_with=engine)
            mapper(Model, model_table)
            df = pd.read_sql(db_session.query(Model).filter(Model.time.between(start_time, end_time)).statement, engine)
            def sum_abs(a):
                return np.sum(abs(a))
            new_df = pd.pivot_table(df, values="value", index="item", aggfunc=sum_abs)
            new_df.sort_values("value", ascending=False, inplace=True)
            new_df["value"] = new_df["value"] / sum(new_df["value"]) * 100
            result = None
            if args['count'] is not None:
                result = new_df.iloc[:int(args['count']), :]
                result.loc['others', 'value'] = 100 - sum(result['value'])
            else:
                result = new_df.iloc[:, :]
            return result['value'].to_dict()
        except Exception as e:
            db_session.rollback()
            abort(404, message=traceback.format_exc())


api.add_resource(ModelAPI, '/Model/<name>')
api.add_resource(ModelStatisticsAPI, '/Model/<name>/statistics')
api.add_resource(TargetAPI, '/Target/<name>')
api.add_resource(TargetStatisticsAPI, '/Target/<name>/statistics')
api.add_resource(TagValuesAPI, '/TagValues')
api.add_resource(ContributeAPI, '/Model/<name>/contribute')

if __name__ == '__main__':
    app.run(host=config['web_api_setting']['host'], port=config['web_api_setting']['port'])

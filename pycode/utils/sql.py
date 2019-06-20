import datetime as dt
import pandas as pd
import logging
from collections import Iterable
from sqlalchemy import MetaData, Table, Column, Integer, DateTime, FLOAT, VARCHAR
from sqlalchemy.orm import relationship, create_session, mapper, scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mssql import BIT, UNIQUEIDENTIFIER
from pycode.utils.load import load_config, load_data

logger = logging.getLogger(__name__)


class TellerSQL:

    def __init__(self, config, engine):
        self.config = config
        self.engine = engine
        Session = scoped_session(sessionmaker(bind=self.engine))
        self.session = Session()
        self.metadata = MetaData(bind=self.engine)
        self.table_list = list()
        self.metadata.reflect(engine)
        for t in self.metadata.sorted_tables:
            self.table_list.append(t.name)

    def create_model_table(self, table_dict):

        if len(table_dict) == 0:
            return

        for table in table_dict.keys():

            # create predict columns
            table_type = table_dict[table]

            if table_type == 'model':
                Table(table, self.metadata,
                      Column('id', Integer, primary_key=True, autoincrement=True),
                      Column('time', DateTime), Column('timestamp', Integer),
                      Column('dqix', FLOAT), Column('ma_dist', FLOAT), Column('mae', FLOAT),
                      Column('conf_idx', FLOAT), Column('predict', FLOAT), Column('intercept', FLOAT))

            if table_type == 'algo':
                Table(table, self.metadata,
                      Column('id', Integer, primary_key=True, autoincrement=True),
                      Column('time', DateTime), Column('timestamp', Integer),
                      Column('algo_name', VARCHAR(100)), Column('coef', FLOAT), Column('predict', FLOAT))

            if table_type == 'var':
                Table(table, self.metadata,
                      Column('id', Integer, primary_key=True), Column('item', VARCHAR(500), primary_key=True),
                      Column('time', DateTime), Column('timestamp', Integer), Column('value', FLOAT))

            if table_type == 'mdist':
                Table(table, self.metadata,
                      Column('id', Integer, primary_key=True), Column('item', VARCHAR(500), primary_key=True),
                      Column('time', DateTime), Column('timestamp', Integer), Column('value', FLOAT))

        self.metadata.create_all()
        logger.info('Create model table')

    def check_model_table(self):

        need_tables = dict()

        for key in self.config['predict_items'].keys():
            need_tables['Model_' + key] = 'model'
            need_tables['Model_' + key + '_algo'] = 'algo'
            need_tables['Model_' + key + '_var'] = 'var'
            need_tables['Model_' + key + '_mdist'] = 'mdist'

        for t in self.table_list:
            if t in need_tables:
                del need_tables[t]

        return need_tables

    def transform_algo(self, model_name):

        model = self.table_mapper('Model_' + model_name)
        algo_model = self.table_mapper('Model_' + model_name + '_algo')
        predict_table = self.session.query(model).all()
        algo_table = self.session.query(algo_model).all()
        algo_list = [name.split('.')[0] for name in self.config['predict_items'][model_name]['algo_name']]

        if len(algo_table) == 0 and len(predict_table) > 0:
            bulk_list = list()
            for row in predict_table:
                for algo_name in algo_list:
                    m = algo_model()
                    m.id = row.id
                    m.time = row.time
                    m.timestamp = row.timestamp
                    m.algo_name = algo_name
                    m.coef = getattr(row, 'coef_' + algo_name)
                    m.predict = getattr(row, 'predict_' + algo_name)
                    bulk_list.append(m)

            self.session.add_all(bulk_list)
            self.session.commit()
            self.session.close()

    def save_result(self, teller):

        class Model(object):
            pass

        main_table_name = 'Model_' + teller.name
        algo_table_name = 'Model_' + teller.name + '_algo'
        var_table_name = 'Model_' + teller.name + '_var'
        mdist_table_name = 'Model_' + teller.name + '_mdist'

        model_table = Table(main_table_name, self.metadata, autoload=True, autoload_with=self.engine)
        mapper(Model, model_table)

        m = Model()
        m.time = teller.time
        m.timestamp = int((m.time - dt.timedelta(hours=8)
                        - dt.datetime(1970, 1, 1)).total_seconds())

        m.ma_dist = teller.MD
        m.dqix = teller.DQIX
        m.conf_idx = teller.RI
        m.intercept = teller.intercept
        m.mae = teller.mae
        m.predict = teller.predict

        logger.info('Save predict result')

        try:
            self.session.add(m)
            self.session.flush()
            main_id = m.id

            # save item
            self._save_result_algo_(algo_table_name, teller.coefs, teller.predicts, main_id, teller.time, m.timestamp)
            self._save_result_info_(var_table_name, teller.X, main_id, teller.time, m.timestamp)
            self._save_result_info_(mdist_table_name, teller.contribute, main_id, teller.time, m.timestamp)

            self.session.commit()
            self.session.close()
        except Exception as e:
            logger.warning(str(e))
            self.session.rollback()

    def _save_result_algo_(self, table_name, coef, predict, main_id, time, timestamp):

        class Model(object):
            pass

        model_table = Table(table_name, self.metadata, autoload=True, autoload_with=self.engine)
        mapper(Model, model_table)

        bulk_list = list()

        for k in coef.keys():
            m = Model()
            m.id = main_id
            m.time = time
            m.timestamp = timestamp
            m.algo_name = k
            m.coef = coef[k]
            m.predict = predict[k]
            bulk_list.append(m)

        self.session.add_all(bulk_list)

    def _save_result_info_(self, table_name, data, main_id, time, timestamp):

        class Model(object):
            pass

        model_table = Table(table_name, self.metadata, autoload=True, autoload_with=self.engine)
        mapper(Model, model_table)

        bulk_list = list()

        for col in data.columns:
            m = Model()
            m.id = main_id
            m.time = time
            m.timestamp = timestamp
            m.item = col
            m.value = data[col].values[0]
            bulk_list.append(m)

        self.session.add_all(bulk_list)

    def table_mapper(self, table_name):

        class TableObj(object):
            pass

        table = Table(table_name, self.metadata, autoload=True, autoload_with=self.engine)
        mapper(TableObj, table)
        return TableObj

    def query_to_dict(self, result):
        data = list()
        for row in result:
            record = row.__dict__
            del record['_sa_instance_state']
            data.append(record)
        return data

    def get_model_result(self, model_name, start_time, end_time):
        try:
            model = self.table_mapper('Model_' + model_name)
            result = self.session.query(model).filter(model.time.between(start_time, end_time))
            data = self.query_to_dict(result)
            df = pd.DataFrame.from_records(data)
            logger.info("Get predict result. model: {0}, start_time: {1}, end_time: {2}".format(model_name, start_time, end_time))
        except Exception as e:
            logger.warning(str(e))
            self.session.rollback()
        return df


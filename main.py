import os
import time
import traceback
import logging, logging.handlers
from threading import Thread
from datetime import datetime as dt
from datetime import timedelta
from pycode.utils.load import load_config, load_data, load_model
from pycode.utils.validation import check_resources
from pycode.feature.core.reader import FeatureReader
from pycode.tellers.ft import FortuneTeller
from pycode.utils.sql import TellerSQL
from pycode.lims.lims import Lims
from pycode.dataport.data_manager import DataManager
from sqlalchemy import create_engine


class MainThread(Thread):

    def __init__(self, sleep_time, teller, teller_sql):
        Thread.__init__(self, name=teller.name)
        self.sleep_time = sleep_time
        self.teller = teller
        self.teller_sql = teller_sql
        self.logger = logging.getLogger(__name__)
        self.index = 0

    def run(self):
        flag = False
        while True:
            try:
                now = dt.now()
                self.logger.info("Predict Start Time: {0}".format(now))
                if flag is False:
                    self.teller.setStartTime(now)
                    flag = True
                self.teller.tell(now)
                self.logger.info("Predict Done Time: {0}".format(now))
                self.teller_sql.save_result(self.teller)
                time.sleep(self.sleep_time)
            except Exception as e:
                self.logger.critical(traceback.format_exc())


# Load config file and check resources
config = load_config("config.yaml")
config = check_resources(config)

# create datamanager
data_manager = DataManager(config)

# Create engine
ft_engine = create_engine(config['sql_connect']['fpc_ft'])


# create model table
teller_sql = TellerSQL(config, ft_engine)
lacking_tables = teller_sql.check_model_table()
if len(lacking_tables) > 0:
    teller_sql.create_model_table(lacking_tables)

for name in config['predict_items'].keys():
    teller_sql.transform_algo(name)

# Logging
log_dir = 'log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

root = logging.getLogger()
root.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler(log_dir + 'debug.log', maxBytes=1000000000, backupCount=10)
fh.setLevel(logging.DEBUG)
fh2 = logging.handlers.RotatingFileHandler(log_dir + 'info_only.log', maxBytes=1000000000, backupCount=10)
fh2.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s - %(threadName)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
fh2.setFormatter(formatter)
root.addHandler(fh)
root.addHandler(fh2)

# build thread_list
thread_list = list()
for name in config['predict_items'].keys():
    teller_sql = TellerSQL(config, ft_engine)  # for multithread
    teller = FortuneTeller(name, data_manager, teller_sql)
    try:
        teller.coefs, teller.intercept = teller.initialModelCoefs()
    except Exception as e:
        logging.getLogger(__name__).error(traceback.format_exc())
        pass
    thread_list.append(MainThread(config['predict_items'][name]['predict_sleep_seconds'],
                                  teller, teller_sql))

# run thread
for t in thread_list:
    t.start()

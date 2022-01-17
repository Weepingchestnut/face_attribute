import logging
import os
import sys
from datetime import datetime


def logger_init(log_file_name='log_test', log_level=logging.DEBUG, log_dir='./logs/', only_file=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.now())[:10] + '.log')
    # formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    formatter = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    if only_file:
        logging.basicConfig(filename=log_path,
                            level=log_level,
                            format=formatter,
                            datefmt='%Y-%d-%m %H:%M:%S')
    else:
        logging.basicConfig(level=log_level,
                            format=formatter,
                            datefmt='%Y-%d-%m %H:%M:%S',
                            handlers=[logging.FileHandler(log_path, mode='a'),
                                      logging.StreamHandler(sys.stdout)])

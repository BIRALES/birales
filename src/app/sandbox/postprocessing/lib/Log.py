import logging as log
from logging.config import fileConfig


class Log:
    LOG_CONFIG = '../config/log.ini'

    def __init__(self):
        fileConfig(self.LOG_CONFIG)

    @staticmethod
    def get_logger():
        return log.getLogger()

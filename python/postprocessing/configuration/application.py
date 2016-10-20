import os
import logging as log

from logging.config import fileConfig
from ConfigParser import ConfigParser


class ApplicationConfiguration:
    config_parser = {}

    def __init__(self, file_name='configuration/local.ini'):
        self.config_parser = ConfigParser()

        if os.path.exists(file_name):
            self.config_parser.read(file_name)
        else:
            raise IOError('Configuration file was not found at %s', file_name)

        # Initialise the logging configuration
        self._init_logger(self.config_parser.get('log', 'FILE_CONFIG'))

        # Inform the user which configurations are going to be used
        log.info('Using application configuration at %s', file_name)
        log.info('Using logger configuration at %s', self.config_parser.get('log', 'FILE_CONFIG'))

    def get(self, section, attribute):
        return self.config_parser.get(section, attribute)

    def get_boolean(self, section, attribute):
        return self.config_parser.getboolean(section, attribute)

    def get_int(self, section, attribute):
        return self.config_parser.getint(section, attribute)

    @staticmethod
    def _init_logger(file_name='configuration/logging.ini'):
        fileConfig(file_name)


config = ApplicationConfiguration()

import logging as log
import os
from ConfigParser import ConfigParser
from logging.config import fileConfig


class ApplicationConfiguration:
    config_parser = {}
    # Set application root
    ROOT = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, file_name='configuration/local.ini'):
        # Set the config parser for the configuration
        self.config_parser = ConfigParser()

        # Initialise the application configuration
        log.info('Using application configuration at %s', file_name)
        self._init_application(file_name=file_name)

        # Initialise the logging configuration
        log.info('Using logger configuration at %s', self.config_parser.get('log', 'FILE_CONFIG'))
        self._init_logger(self.config_parser.get('log', 'FILE_CONFIG'))

    def _init_application(self, file_name='configuration/local.ini'):
        file_path = os.path.join(self.ROOT, file_name)
        if os.path.exists(file_path):
            self.config_parser.read(file_path)
        else:
            raise IOError('Application configuration file was not found at %s', file_path)

    def _init_logger(self, file_name='configuration/logging.ini'):
        file_path = os.path.join(self.ROOT, file_name)
        if os.path.exists(file_path):
            fileConfig(file_path)
        else:
            raise IOError('Configuration file was not found at %s', file_path)

    def get(self, section, attribute):
        return self.config_parser.get(section, attribute)

    def get_boolean(self, section, attribute):
        return self.config_parser.getboolean(section, attribute)

    def get_int(self, section, attribute):
        return self.config_parser.getint(section, attribute)


config = ApplicationConfiguration()

import logging as log
import os
from ConfigParser import ConfigParser
from logging.config import fileConfig


class ApplicationConfiguration:
    config_parser = {}
    # Set application root
    ROOT = os.path.dirname(os.path.dirname(__file__))

    env_prefix = 'BIRALES'
    env_glue = '__'

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
        return self.get_env(section, attribute) or self.config_parser.get(section, attribute)

    def get_boolean(self, section, attribute):
        env = self.get_env(section, attribute)
        if env:
            return bool(env)
        return self.config_parser.getboolean(section, attribute)

    def get_int(self, section, attribute):
        env = self.get_env(section, attribute)
        if env:
            return int(env)

        return self.config_parser.getint(section, attribute)

    def get_int_list(self, section, attribute):
        """
        Reads the corresponding 'attribute' parameter in the 'section' section from the INI
        configuration file and returns a list of integers

        :param section: The section fo the parameter
        :param attribute: The parameter name
        :return: A list of integers or and empty list
        """
        config_entry = self.config_parser.get(section, attribute)
        env = self.get_env(section, attribute)
        if env:
            config_entry = env

        try:
            return [int(n) for n in config_entry.split(',')]
        except ValueError:
            # Return an empty array if the array is not valid
            return []

    def get_env(self, section, attribute):
        env_variable_key = self.env_glue.join([self.env_prefix, section, attribute]).upper()

        if env_variable_key in os.environ:
            return os.environ[env_variable_key]


config = ApplicationConfiguration()

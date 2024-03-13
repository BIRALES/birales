import ast
import configparser
import datetime
import logging
import logging as log
import logging.config as log_config
import os
import re
import glob
from logging.handlers import TimedRotatingFileHandler

from mongoengine import connect
from mongoengine.base.datastructures import BaseList

from pybirales import settings


class BiralesConfig:
    def __init__(self, config_file_path=None, config_options=None):
        """
        Initialise the BIRALES configuration

        :param config_file_path: The path to the BIRALES configuration file
        :param config_options: Configuration options to override the default config settings
        :return:
        """

        # Set configuration files root
        if "BIRALES_CONFIG_DIRECTORY" not in os.environ:
            logging.error("BIRALES_CONFIG_DIRECTORY not defined, cannot configure")
            return

        # Set configuration root directory
        self._config_root = os.path.expanduser(os.environ['BIRALES_CONFIG_DIRECTORY'])

        # Check if file path is valid
        if not os.path.exists(self._config_root) or not os.path.isdir(self._config_root):
            logging.error("BIRALES_CONFIG_DIRECTORY is invalid ({}), path does not exist".format(self._config_root))
            return

        # Temporary settings directory
        self._loaded_settings = {}

        # The configuration parser of the BIRALES system
        self._parser = configparser.RawConfigParser()

        # Load the logging configuration
        try:
            self.log_filepath = self._set_logging_config(config_options['observation']['name'])
        except KeyError:
            self.log_filepath = self._set_logging_config(
                'BIRALES_observation_' + datetime.datetime.utcnow().isoformat('T'))
        except TypeError:
            self.log_filepath = self._set_logging_config(
                'BIRALES_observation_' + datetime.datetime.utcnow().isoformat('T'))

        # Load the instrument config file
        self._load_from_file(os.path.join(self._config_root, 'instrument.ini'))

        # Load pointing config files
        for f in glob.glob(os.path.join(self._config_root, 'pointing/*.ini')):
            self._load_from_file(f)

        if config_file_path:
            # Set the configurations from file (can be multiple files)
            if type(config_file_path) in [list, tuple, BaseList]:
                for config_file in config_file_path:
                    self._load_from_file(config_file)
            else:
                self._load_from_file(config_file_path)
        else:
            # load the birales default configuration
            self._load_from_file(os.path.join(self._config_root, 'birales.ini'))

        # Override the configuration with settings passed on in the config_options dictionary
        self.update_config(config_options)

        # Connect to database if enabled
        if settings.database.load_database:
            self._db_connect()

    def get_root_path(self):
        """ Return configuration root path """
        return self._config_root

    def _load_from_file(self, config_filepath):
        """
        Load the configuration of the BIRALES application into the settings.py file

        :param config_filepath: The path to the configuration file
        :return: None
        """

        # If an absolute path was provided, use config_filepath directly,
        # Otherwise assume that the file is in the birales config directory
        if not os.path.isabs(config_filepath):
            config_filepath = os.path.join(self._config_root, config_filepath)

        # Load the configuration file requested by the user
        try:
            log.info('Loading the {} configuration file.'.format(config_filepath))
            with open(os.path.expanduser(config_filepath)) as f:
                self._parser.read_file(f)
                log.info('Loaded the {} configuration file.'.format(config_filepath))
        except IOError:
            log.warning('Config file at {} was not found'.format(config_filepath))
        except configparser.Error:
            log.exception('An error has occurred whilst parsing configuration file at: %s', config_filepath)

        # Once config file is read, reload configuration
        self.load()

    def update_config(self, config_options):
        """
        Override the configuration settings using an external dictionary

        :param config_options:
        :return:
        """
        # config has to be loaded before it can be updated
        self.load()

        if not config_options:
            return None

        for section in config_options:
            if isinstance(config_options[section], dict):
                # If settings is a dictionary, add it as a section
                for (key, value) in config_options[section].items():
                    self._parser.set(section, key, value)
            else:
                # Else, put the configuration in the observations settings
                self._parser.set('observation', section, config_options[section])

        # Re-load the system configuration upon initialisation
        self.load()

    def _set_logging_config(self, observation_name):
        """
        Load the logging configuration
        :return:
        """

        # Load the logging configuration file
        config_filepath = os.path.join(self._config_root, 'logging.ini')
        self._load_from_file(config_filepath)
        log_config.fileConfig(config_filepath, disable_existing_loggers=False)

        # Override the logger's debug level
        log.getLogger().setLevel('INFO')

        # Create directory for file log
        directory = os.path.join('/var/log/birales', '{:%Y_%m_%d}'.format(datetime.datetime.now()))
        if not os.path.exists(directory):
            os.makedirs(directory)

        log_path = os.path.join(directory, observation_name + '.log')

        handler = TimedRotatingFileHandler(log_path, when="h", interval=1, backupCount=0, utc=True)
        formatter = log.Formatter(self._parser.get('formatter_formatter', 'format'))
        handler.setFormatter(formatter)
        log.getLogger().addHandler(handler)

        return log_path

    def get(self, section, key):
        return self._parser.get(section, key)

    def _db_connect(self):
        """
        Connect to the database using the loaded settings file

        :return:
        """
        username = os.environ['BIRALES__DB_USERNAME']
        password = os.environ['BIRALES__DB_PASSWORD']

        if username is None or password is None:
            raise ConnectionAbortedError(f"Could not connect to DB. BIRALES__DB_USERNAME ({username}) "
                                         f"or BIRALES__DB_PASSWORD ({password}) environment variables not set")

        if settings.database.authentication:
            self.db_connection = connect(
                db=settings.database.name,
                username=username,
                password=password,
                port=settings.database.port,
                host=settings.database.host,
                authentication_source="birales")
        else:
            self.db_connection = connect(settings.database.host)

        log.info('Successfully connected to the {} database'.format(settings.database.name))

    def load(self):
        """
        Use a config parser to build the settings module. This module is accessible through
        the application

        :return:
        """

        # Temporary class to create section object in settings file
        class Section(object):
            def settings(self):
                return self.__dict__.keys()

        # Loop over all sections in config file
        for section in self._parser.sections():
            # Create instance to inject into settings file
            instance = Section()

            for (k, v) in self._parser.items(section):
                # If value is a string, interpret it
                if isinstance(v, str):
                    # Check if value is a number, boolean, list or string
                    if re.match(re.compile("^True|False|[0-9]+(\.[0-9]*)?$"), v) or \
                       re.match(r"^\[.*\]$", re.sub(r'\s+', '', v)) or \
                       re.match(r"^\{.*\}$", re.sub(r'\s+', '', v)):
                        # Evaluate value as an AST literal
                        setattr(instance, k, ast.literal_eval(v))

                    # Otherwise use the value as is (string)
                    else:
                        setattr(instance, k, v)
                else:
                    setattr(instance, k, v)

            # Add object instance to settings
            setattr(settings, section, instance)

        log.info('Configurations successfully loaded.')

    @staticmethod
    def to_dict():
        """ Return the dictionary representation of the Birales configuration
        :return:
        """

        return {section: settings.__dict__[section].__dict__ for section in settings.__dict__.keys() if
                not section.startswith('__') and settings.__dict__[section] is not None}


if __name__ == "__main__":
    BiralesConfig(config_file_path=['birales.ini', 'test_overload.ini'])
    logging.info("All done")
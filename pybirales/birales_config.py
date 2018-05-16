import ast
import datetime
import logging as log
import logging.config as log_config
import os
import re
from logging.handlers import TimedRotatingFileHandler

import configparser
from mongoengine import connect

from pybirales import settings


class BiralesConfig:
    def __init__(self, config_file_path=None, config_options=None):
        """
        Initialise the BIRALES configuration

        :param observation_name: The name of the observations
        :param config_file_path: The path to the BIRALES configuration file
        :param config_options: Configuration options to override the default config settings
        :return:
        """
        # Specify whether the configuration settings were loaded in the settings.py package
        self._loaded = False

        # The configuration parser of the BIRALES system
        self._parser = configparser.RawConfigParser()

        # Load the logging configuration
        try:
            self._set_logging_config(config_options['observation']['name'])
        except KeyError:
            self._set_logging_config('BIRALES_observation_' + datetime.datetime.utcnow().isoformat('T'))
        except TypeError:
            self._set_logging_config('BIRALES_observation_' + datetime.datetime.utcnow().isoformat('T'))

        if config_file_path:
            # Set the configurations from file (can be multiple files)
            log.info('Loading configuration files')
            for config_file in config_file_path:
                self._load_from_file(config_file)
        else:
            # throw an error - no configuration could be loaded
            pass

        # Load the ROACH backend settings
        backend_path = os.path.join(os.path.dirname(__file__), self._parser.get('receiver', 'backend_config_filepath'))
        self._load_from_file(backend_path)

        if config_options:
            # Override the configuration with settings passed on in the config_options dictionary
            self.update_config(config_options)

    def is_loaded(self):
        """

        :return:
        """
        return self._loaded

    def _load_from_file(self, config_filepath):
        """
        Load the configuration of the BIRALES application into the settings.py file

        :param config_filepath: The path to the configuration file
        :return: None
        """

        # Load the configuration file requested by the user
        try:
            log.info('Loading the {} configuration file.'.format(config_filepath))
            with open(os.path.expanduser(config_filepath)) as f:
                self._parser.read_file(f)
                log.info('Loaded the {} configuration file.'.format(config_filepath))
        except IOError:
            log.info('Config file at {} was not found'.format(config_filepath))
        except configparser.Error:
            log.exception('An error has occurred whilst parsing configuration file at: %s', config_filepath)

    def update_config(self, config_options):
        """
        Override the configuration settings using an external dictionary

        :param config_options:
        :return:
        """

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
        config_filepath = os.path.join(os.path.dirname(__file__), 'configuration/logging.ini')
        self._load_from_file(config_filepath)
        log_config.fileConfig(config_filepath, disable_existing_loggers=False)

        # Override the logger's debug level
        log.getLogger().setLevel('DEBUG')

        # Create directory for file log
        directory = os.path.join('/var/log/birales', '{:%Y_%m_%d}'.format(datetime.datetime.now()))
        if not os.path.exists(directory):
            os.makedirs(directory)

        log_path = os.path.join(directory, observation_name + '.log')

        handler = TimedRotatingFileHandler(log_path, when="h", interval=1, backupCount=5, utc=True)
        formatter = log.Formatter(self._parser.get('formatter_formatter', 'format'))
        handler.setFormatter(formatter)
        log.getLogger().addHandler(handler)

    @staticmethod
    def _db_connect():
        """
        Connect to the database using the loaded settings file

        :return:
        """

        if settings.database.authentication:
            connect(
                db=settings.database.name,
                username=settings.database.user,
                password=settings.database.password,
                port=settings.database.port,
                host=settings.database.host)
        else:
            connect(settings.database.host)

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
                if isinstance(v, basestring):
                    # Check if value is a number of boolean
                    if re.match(re.compile("^True|False|[0-9]+(\.[0-9]*)?$"), v) is not None:
                        setattr(instance, k, ast.literal_eval(v))

                    # Check if value is a list
                    elif re.match("^\[.*\]$", re.sub('\s+', '', v)):
                        setattr(instance, k, ast.literal_eval(v))

                    # Otherwise it is a string
                    else:
                        setattr(instance, k, v)
                else:
                    setattr(instance, k, v)

            # Add object instance to settings
            setattr(settings, section, instance)

        log.info('Configurations successfully loaded.')

        if not self.is_loaded():
            # Connect to the database
            self._db_connect()

        self._loaded = True
        # todo - Validate the loaded configuration file

    @staticmethod
    def to_dict():
        """
        Return the dictionary representation of the Birales configuration

        :return:
        """
        return {section: settings.__dict__[section].__dict__ for section in settings.__dict__.keys() if
                not section.startswith('__')}

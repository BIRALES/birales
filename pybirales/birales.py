from pybirales.repository.models import Observation
import ast
import configparser
import datetime
import logging as log
import logging.config as log_config
import os
import re
from pybirales import settings
from mongoengine import connect
from pybirales.services.calibration.calibration import CalibrationFacade
from pybirales.app.app import run


class BiralesConfig:
    LOCAL_CONFIG = '~/.birales/local.ini'

    def __init__(self, config_file_path, config_options=None):
        self._parser = None

        self._load_from_file(config_file_path)

        self.update_config(config_options)

        self._loaded = False

    def is_loaded(self):
        return self._loaded

    def _load_from_file(self, config_file):
        """
        Load the configuration of the BIRALES application into the settings.py file

        :param config_file: The path to the application configuration file
        :return: None
        """

        parser = configparser.RawConfigParser()

        # Load the configuration file requested by the user
        with open(config_file) as f:
            parser.read_file(f)
            log_config.fileConfig(config_file, disable_existing_loggers=False)
            log.info('Loading configuration file at {}.'.format(config_file))

        # Override the default configuration file with the ones specified in the local.ini
        try:
            with open(os.path.expanduser(BiralesConfig.LOCAL_CONFIG)) as f:
                parser.read_file(f)
                log.info('Loading local configuration file at {}.'.format(BiralesConfig.LOCAL_CONFIG))
        except IOError:
            log.info(
                'Local config file not found in {}. Using the default configuration.'.format(
                    BiralesConfig.LOCAL_CONFIG))

        self._parser = parser

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
                # Else, put the configuration in the observation settings
                self._parser.set('observation', section, config_options[section])

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
                # Check if value is a number of boolean

                # If value is a string, interpret it
                if isinstance(v, basestring):
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


class BiralesFacade:
    def __init__(self, configuration):
        # The configuration associated with this facade Instance
        self.configuration = configuration

        # Load the system configuration upon initialisation
        self.configuration.load()

        self._pipeline_manager = None

        # Ensure that the system was initialised correctly
        self.validate_init()

    def validate_init(self):
        pass

    def start_observation(self, pipeline_manager):
        """
        Start the observation

        :param pipeline_manager: The pipeline manager associated with this observation
        :return:
        """
        # Ensure status of the pipeline is correct
        # Check if calibration is required
        # Point the telescope
        # Start the chosen pipeline

        if pipeline_manager:
            observation = Observation(name=settings.observation.name,
                                      date_time_start=datetime.datetime.utcnow(),
                                      settings=self.configuration.to_dict())
            observation.save()

            self.configuration.update_config({'observation': {'id': observation.id}})
            # Re-load the system configuration upon initialisation
            self.configuration.load()

            pipeline_manager.start_pipeline(settings.observation.duration)

            observation.date_time_end = datetime.datetime.utcnow()
            observation.save()

    def build_pipeline(self, pipeline_builder):
        """

        :param pipeline_builder:
        :return: None
        """

        log.info('Building the {} Manager.'.format(pipeline_builder.manager.name))

        pipeline_builder.build()

        self._pipeline_manager = pipeline_builder.manager

        log.info('{} initialised successfully.'.format(self._pipeline_manager.name))

        return pipeline_builder.manager

    def calibrate(self):
        """

        :return:
        """

        cf = CalibrationFacade()
        cf.calibrate()


    def start_server(self, configuration):
        run(configuration)
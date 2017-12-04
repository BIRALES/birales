from pybirales.services.calibrator.facade import CalibrationFacade

import ast
import configparser
import logging as log
import logging.config as log_config
import os
import re
from pybirales import settings


class BiralesConfig:
    LOCAL_CONFIG = '~/.birales.ini'

    def __init__(self, config_file_path, config_options):
        self._parser = None

        self._load_from_file(config_file_path)

        self.update_config(config_options)

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
                'Local config file not found in {}. Using the default configuration.'.format(BiralesConfig.LOCAL_CONFIG))

        self._parser = parser

    def update_config(self, config_options):
        """
        Override the configuration settings using an external dictionary

        :param config_options:
        :return:
        """

        for section in config_options:
            for (key, value) in config_options[section].items():
                self._parser.set(section, key, value)

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

        # todo - Validate the loaded configuration file


class BiralesFacade:
    def __init__(self, configuration):
        # Load the system configuration upon initialisation
        configuration.load()

        # Initialise the facades of the application
        self._frontend_subsystem = None
        self._calibration_subsystem = CalibrationFacade()

        self._pipeline_manager = None

        # Ensure that the system was initialised correctly
        self.validate_init()

    def validate_init(self):
        pass

    def run_pipeline(self):
        """

        :return:
        """

        # Ensure status of the pipeline is correct
        # Check if calibration is required
        # Point the telescope
        # Start the chosen pipeline
        if self._pipeline_manager:
            self._pipeline_manager.start()

    def build_pipeline(self, pipeline_builder):
        """

        :param pipeline_builder:
        :return: None
        """
        log.info('Building the {} Manager.'.format(pipeline_builder.manager.name))

        pipeline_builder.build()

        self._pipeline_manager = pipeline_builder.manager

        log.info('{} initialised successfully.'.format(self._pipeline_manager.name))

    @staticmethod
    def start_server(flask_app):
        """

        :param flask_app: The flask application
        :return:
        """

        flask_app.run(flask_app)

    @staticmethod
    def calibrate():
        """
        Calibrate the instrument. Run the calibration Routine.

        :return:
        """
        pass

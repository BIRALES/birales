from pybirales.services.calibrator.facade import CalibrationFacade

import ast
import configparser
import logging as log
import os
import re
import settings


class BiralesFacade:
    LOCAL_CONFIG = 'configuration/local.ini'

    def __init__(self, config_file):
        self._load_configuration(config_file)

        # Initialise the facades of the application
        self._frontend_subsystem = None
        self._calibration_subsystem = CalibrationFacade()

        self._pipeline_manager = None

        log.info('BIRALES Initialised')

    def _load_configuration(self, config_file):
        """
        Load the configuration of the BIRALES application into the settings.py file

        :param config_file: The path to the application configuration file
        :return: None
        """

        parser = configparser.RawConfigParser()

        # Load the configuration file requested by the user
        with open(config_file) as f:
            parser.read_file(f)
            log.config.fileConfig(config_file, disable_existing_loggers=False)
            log.info('Loading configuration file at {}.'.format(config_file))

        # Override the default configuration file with the ones specified in the local.ini
        try:
            with open(os.path.join(os.path.dirname(__file__), BiralesFacade.LOCAL_CONFIG)) as f:
                parser.read_file(f)
                log.info('Loading local configuration file at {}.'.format(BiralesFacade.LOCAL_CONFIG))
        except IOError:
            log.info(
                'Local config file not found in {}. Using default configuration.'.format(BiralesFacade.LOCAL_CONFIG))

        # Temporary class to create section object in settings file
        class Section(object):
            def settings(self):
                return self.__dict__.keys()

        # Loop over all sections in config file
        for section in parser.sections():
            # Create instance to inject into settings file
            instance = Section()

            for (k, v) in parser.items(section):
                # Check if value is a number of boolean
                if re.match(re.compile("^True|False|[0-9]+(\.[0-9]*)?$"), v) is not None:
                    setattr(instance, k, ast.literal_eval(v))

                # Check if value is a list
                elif re.match("^\[.*\]$", re.sub('\s+', '', v)):
                    setattr(instance, k, ast.literal_eval(v))

                # Otherwise it is a string
                else:
                    setattr(instance, k, v)

            # Add object instance to settings
            setattr(settings, section, instance)

        log.info('Configurations successfully loaded.')

        # todo - Validate the loaded configuration file

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

        pipeline_builder.build()

        self._pipeline_manager = pipeline_builder.manager

    @staticmethod
    def calibrate(self):
        """
        Calibrate the instrument. Run the calibration Routine.

        :return:
        """
        pass

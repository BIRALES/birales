import ast
import datetime
import logging as log
import logging.config as log_config

from logging.handlers import TimedRotatingFileHandler
import time
import os
import re

import configparser
import signal
from mongoengine import connect

from pybirales import settings
from pybirales.app.app import run
from pybirales.repository.models import Observation
from pybirales.services.calibration.calibration import CalibrationFacade
from pybirales.services.instrument.backend import Backend
from pybirales.services.instrument.best2 import BEST2
from pybirales.listeners.listeners import NotificationsListener


class BiralesConfig:
    LOCAL_CONFIG = os.path.join(os.environ['HOME'], '.birales/local.ini')

    def __init__(self, config_file_path, config_options=None):
        """
        Initialise the BIRALES configuration

        :param config_file_path: The path to the BIRALES configuration file
        :param config_options: Configuration options to override the default config settings
        :return:
        """

        # The configuration parser of the BIRALES system
        self._parser = configparser.RawConfigParser()

        # Set the configurations from file (can be multiple files)
        for config_file in config_file_path:
            self._load_from_file(config_file)

        # Load the ROACH backend settings
        backend_path = os.path.join(os.path.dirname(config_file_path[0]),
                                    self._parser.get('receiver', 'backend_config_filepath'))
        self._load_from_file(backend_path)

        # Override the configuration with settings passed on in the config_options dictionary
        self.update_config(config_options)

        self._set_logging_config(config_file_path)

        # Specify whether the configuration settings were loaded in the settings.py package
        self._loaded = False

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
            with open(config_filepath) as f:
                self._parser.read_file(f)
                log.info('Loaded the {} configuration file.'.format(config_filepath))
        except IOError:
            log.info('Config file at {} was not found'.format(config_filepath))

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

        # Re-load the system configuration upon initialisation
        self.load()

    def _set_logging_config(self, config_filepath):
        """

        :param config_filepath:
        :return:
        """

        # Load the logging configuration file
        log_config.fileConfig(config_filepath, disable_existing_loggers=False)

        # Override the logger's debug level
        log.getLogger().setLevel(self._parser.get('logger_root', 'level'))

        # Create directory for file log
        directory = os.path.join(self._parser.get('handler_rot_handler', 'log_directory'),
                                 '{:%Y_%m_%d}'.format(datetime.datetime.now()))
        if not os.path.exists(directory):
            os.makedirs(directory)

        log_path = os.path.join(directory, self._parser.get('observation', 'name') + '.log')

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

        # Initialise and start the listeners / subscribers of the BIRALES application
        self._listeners = self._init_listeners()

        self._pipeline_manager = None

        self._calibration = CalibrationFacade()

        self._instrument = None

        self._backend = None

        # Set interrupt signal handler if not already set
        if signal.getsignal(signal.SIGINT) == signal.SIG_DFL:
            log.info("Setting signal handler in BiralesFacade")
            signal.signal(signal.SIGINT, self._signal_handler)
        else:
            log.info("Signal handler already set, not setting it in BiralesFacade")

    def _signal_handler(self, signum, frame):
        """ Capturing interrupt signal """
        log.info("Ctrl-C detected by process %s, stopping pipeline", os.getpid())

        self.stop_observation()

    def validate_init(self):
        pass

    def start_observation(self, pipeline_manager):
        """
        Start the observation

        :param pipeline_manager: The pipeline manager associated with this observation
        :return:
        """

        if not settings.manager.offline:

            # Initialisation of the backend system
            self._backend = Backend.Instance()
            time.sleep(1)
            self._backend.start(program_fpga=True, equalize=True, calibrate=True)

            # Point the BEST Antenna
            if settings.instrument.enable_pointing:
                self._instrument = BEST2.Instance()
                self._instrument.move_to_declination(settings.beamformer.reference_declination)

        # Ensure that the status of the Backend/BEST/Pipeline is correct.
        # Perform any necessary checks before starting the pipeline
        self.validate_init()

        # Start the chosen pipeline
        if pipeline_manager:
            observation = Observation(name=settings.observation.name,
                                      date_time_start=datetime.datetime.utcnow(),
                                      settings=self.configuration.to_dict())
            observation.save()

            self.configuration.update_config({'observation': {'id': observation.id}})

            pipeline_manager.start_pipeline(settings.observation.duration)

            observation.date_time_end = datetime.datetime.utcnow()
            observation.save()

        # All done, stop observation
        self.stop_observation()

    def stop_observation(self):
        """ Stop observation """
        # Stop pipeline
        if self._pipeline_manager is not None:
            self._pipeline_manager.stop_pipeline()

        if self._instrument is not None:
            self._instrument.stop_best2_server()

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

    def calibrate(self, correlator_pipeline_manager):
        """
        Calibration routine, which will use the correlator pipeline manager

        :param correlator_pipeline_manager:
        :return:
        """

        self.configuration.update_config({'observation': {'type': 'calibration'}})
        calib_dir, corr_matrix_filepath = self._calibration.get_calibration_filepath()

        self.configuration.update_config({'corrmatrixpersister': {'corr_matrix_filepath': corr_matrix_filepath}})

        if settings.calibration.generate_corrmatrix:
            # Run the correlator pipeline to get model visibilities
            self.start_observation(pipeline_manager=correlator_pipeline_manager)

        log.info('Generating calibration coefficients')
        self._calibration.calibrate(calib_dir, corr_matrix_filepath)

        # Load Coefficients to ROACH
        if self._backend:
            self._backend.load_calibration_coefficients(amplitude=self._calibration.dict_real,
                                                        phase=self._calibration.dict_imag)
            log.info('Calibration coefficients loaded to the ROACH')

    @staticmethod
    def start_server():
        run()

    @staticmethod
    def _init_listeners():
        listeners = []
        if settings.observation.notifications:
            listeners.append(NotificationsListener())

        # Start the listener threads
        for l in listeners:
            l.start()

        return listeners

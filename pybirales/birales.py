import ast
import datetime
import logging as log
import logging.config as log_config
import os
import re
import signal
import time
from logging.handlers import TimedRotatingFileHandler

import configparser
from mongoengine import connect

from pybirales import settings
from pybirales.app.app import run
from pybirales.events.events import ObservationStartedEvent, ObservationFinishedEvent
from pybirales.events.publisher import EventsPublisher
from pybirales.listeners.listeners import NotificationsListener
from pybirales.pipeline.pipeline import get_builder_by_id
from pybirales.repository.models import Observation
from pybirales.services.calibration.calibration import CalibrationFacade
from pybirales.services.instrument.backend import Backend
from pybirales.services.instrument.best2 import BEST2
from pybirales.services.scheduler.exceptions import SchedulerException, NoObservationsQueuedException
from pybirales.services.scheduler.observation import ScheduledObservation, ScheduledCalibrationObservation
from pybirales.services.scheduler.scheduler import ObservationsScheduler


class BiralesConfig:
    def __init__(self, config_file_path, config_options=None):
        """
        Initialise the BIRALES configuration

        :param config_file_path: The path to the BIRALES configuration file
        :param config_options: Configuration options to override the default config settings
        :return:
        """
        # Specify whether the configuration settings were loaded in the settings.py package
        self._loaded = False

        # The configuration parser of the BIRALES system
        self._parser = configparser.RawConfigParser()

        # Set the configurations from file (can be multiple files)
        for config_file in config_file_path:
            self._load_from_file(config_file)

        # Load the ROACH backend settings
        backend_path = os.path.join(os.path.dirname(__file__), self._parser.get('receiver', 'backend_config_filepath'))
        self._load_from_file(backend_path)

        # Override the configuration with settings passed on in the config_options dictionary
        self.update_config(config_options)

        self._set_logging_config(config_file_path)

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

        if not self.is_loaded:
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
        self._listeners = self._get_listeners()

        self._start_listeners()

        self._pipeline_manager = None

        self._calibration = CalibrationFacade()

        self._instrument = None

        self._backend = None

        self._publisher = EventsPublisher.Instance()

        self._scheduler = None

        signal.signal(signal.SIGINT, self._signal_handler)

    def _update_config(self, configuration):
        self.configuration = configuration

        self.configuration.load()

    def _signal_handler(self, signum, frame):
        """ Capturing interrupt signal """
        log.info("Ctrl-C detected by process %s, stopping pipeline", os.getpid())

        self.stop()

    def validate_init(self):
        pass

    def _load_backend(self):
        log.info('Loading Backend')
        # Initialisation of the backend system
        self._backend = Backend.Instance()
        time.sleep(1)
        self._backend.start(program_fpga=True, equalize=True, calibrate=True)

        log.info('Backend loaded')

    def start_observation(self, pipeline_manager):
        """
        Start the observation

        :param pipeline_manager: The pipeline manager associated with this observation
        :return:
        """

        if not settings.manager.offline:
            self._load_backend()

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

            # Fire an Observation was Started Event
            self._publisher.publish(ObservationStartedEvent(observation, pipeline_manager.name))

            self.configuration.update_config({'observation': {'id': observation.id}})

            pipeline_manager.start_pipeline(settings.observation.duration)

            observation.date_time_end = datetime.datetime.utcnow()
            observation.save()

            self._publisher.publish(ObservationFinishedEvent(observation, pipeline_manager.name))

    def stop(self):
        """
        Stop all the BIRALES system sub-modules

        :return:
        """

        # Stop pipeline
        if self._pipeline_manager is not None:
            log.debug('Stopping the pipeline manager')
            self._pipeline_manager.stop_pipeline()

        # Stop the BEST2
        if self._instrument is not None:
            log.debug('Stopping the instrument')
            self._instrument.stop_best2_server()

        # Stop the Observations scheduler
        if self._scheduler is not None:
            log.debug('Stopping the Scheduler instance')
            self._scheduler.stop()

        # Stop the listener threads
        if self._listeners is not None:
            log.debug('Stopping the listeners')
            self._stop_listeners()

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

        if not settings.manager.offline:
            self._load_backend()

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
            log.info('Loading calibration coefficients to the ROACH')
            self._backend.load_calibration_coefficients(amplitude=self._calibration.dict_real,
                                                        phase=self._calibration.dict_imag)
            log.info('Calibration coefficients loaded to the ROACH')
        else:
            log.warning("Could not load calibration coefficients. Backend is offline.")

    def start_scheduler(self, schedule_file_path, file_format):
        """
        Start the scheduler

        :param schedule_file_path:
        :param file_format:
        :return:
        """

        def run_observation(observation):
            """
            Run the pipeline using the provided observation settings

            :param observation: A ScheduledObservation object
            :return:
            """

            # Load the BIRALES configuration from file
            self._update_config(configuration=BiralesConfig(observation.config_file, observation.parameters))

            # Build the pipeline manager
            manager = self.build_pipeline(pipeline_builder=get_builder_by_id(observation.pipeline_name))

            # Start observation
            if isinstance(observation, ScheduledObservation):
                self.start_observation(pipeline_manager=manager)

            # Calibrate the Instrument
            if isinstance(observation, ScheduledCalibrationObservation):
                self.calibrate(correlator_pipeline_manager=manager)

        try:
            # The Scheduler responsible for the scheduling of observations
            self._scheduler = ObservationsScheduler(observation_run_func=run_observation)
            self._scheduler.load_from_file(schedule_file_path, file_format)
            self._scheduler.start()
        except KeyboardInterrupt:
            log.info('Ctrl-C received. Terminating the scheduler process.ws')
            self._scheduler.stop()
        except NoObservationsQueuedException:
            log.info('Could not run Scheduler since no valid observations could be scheduled.')
            self._scheduler.stop()
        except SchedulerException:
            log.exception('A fatal Scheduler error has occurred. Terminating the scheduler process.')
            self._scheduler.stop()
        else:
            log.info('Scheduler finished successfully.')

        self.stop()

    @staticmethod
    def start_server():
        run()

    @staticmethod
    def _get_listeners():
        listeners = []
        if settings.observation.notifications:
            listeners.append(NotificationsListener())

        return listeners

    def _start_listeners(self):
        for l in self._listeners:
            l.start()

    def _stop_listeners(self):
        for l in self._listeners:
            l.stop()

        log.info('Listeners stopped')

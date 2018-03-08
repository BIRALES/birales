import datetime
import logging as log
import os
import signal
import time

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
from pybirales.birales_config import BiralesConfig
from pybirales.pipeline.base.definitions import BEST2PointingException


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

                try:
                    self._instrument.move_to_declination(settings.beamformer.reference_declination)
                except BEST2PointingException:
                    log.warning('Could not point the BEST2 Antenna to DEC: {:0.2f}'.format(
                        settings.beamformer.reference_declination))
            else:
                log.info('BEST-II pointing is disabled as specified in configuration')

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

        # Stop the ROACH backend
        if self._backend is not None:
            log.debug('Stopping the Backend')
            self._backend.stop()

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

    def reset_calibration_coefficients(self):
        """
        Reset calibration coefficients

        :return:
        """

        # Load the backend
        if not settings.manager.offline:
            self._load_backend()

        # Reset calibration coefficients on ROACH
        if self._backend:
            log.info('Resetting the calibration coefficients')
            self._backend.load_calibration_coefficients(amplitude=self._calibration.real_reset_coeffs,
                                                        phase=self._calibration.imag_reset_coeffs)

    def calibrate(self, correlator_pipeline_manager):
        """
        Calibration routine, which will use the correlator pipeline manager

        :param correlator_pipeline_manager:
        :return:
        """

        if not settings.manager.offline:
            self._load_backend()

        # Reset calibration coefficients before applying new ones
        self.reset_calibration_coefficients()

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
                if observation.is_calibration_obs:
                    self.calibrate(correlator_pipeline_manager=manager)
                else:
                    self.start_observation(pipeline_manager=manager)

            # Calibrate the Instrument
            if isinstance(observation, ScheduledCalibrationObservation):
                self.calibrate(correlator_pipeline_manager=manager)

        try:
            # The Scheduler responsible for the scheduling of observations
            self._scheduler = ObservationsScheduler(observation_run_func=run_observation)

            if schedule_file_path:
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

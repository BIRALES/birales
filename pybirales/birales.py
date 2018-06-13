import datetime
import logging as log
import os
import signal
import time

from pybirales import settings
from pybirales.birales_config import BiralesConfig
from pybirales.events.events import ObservationStartedEvent, ObservationFinishedEvent
from pybirales.events.publisher import EventsPublisher
from pybirales.listeners.listeners import NotificationsListener
from pybirales.pipeline.base.definitions import BEST2PointingException
from pybirales.pipeline.pipeline import get_builder_by_id
from pybirales.repository.models import Observation
from pybirales.services.calibration.calibration import CalibrationFacade
from pybirales.services.instrument.backend import Backend
from pybirales.services.instrument.best2 import BEST2
from pybirales.services.post_processing.processor import PostProcessor
from pybirales.services.scheduler.exceptions import SchedulerException, NoObservationsQueuedException
from pybirales.services.scheduler.observation import ScheduledObservation, ScheduledCalibrationObservation
from pybirales.services.scheduler.scheduler import ObservationsScheduler


class BiralesFacade:
    def __init__(self, configuration):
        # The configuration associated with this facade Instance
        self.configuration = configuration

        # Load the system configuration upon initialisation
        self.configuration.load()

        # Initialise and start the listeners / subscribers of the BIRALES application
        self._listeners = self._get_listeners()

        self._start_listeners()

        self._instrument = None

        self._backend = None

        self._scheduler = None

        signal.signal(signal.SIGINT, self._signal_handler)

    def _update_config(self, configuration):
        self.configuration = configuration

        self.configuration.load()

    def _signal_handler(self, signum, frame):
        """ Capturing interrupt signal """
        log.info("Ctrl-C detected by process %s, stopping pipeline", os.getpid())

        self.stop()

    def _load_backend(self):
        log.info('Loading Backend')
        # Initialisation of the backend system
        self._backend = Backend.Instance()
        time.sleep(1)
        self._backend.start(program_fpga=True, equalize=True, calibrate=True)

        log.info('Backend loaded')

    def run_observation(self, observation):
        """
        Start the observation

        :return:
        """

        if not settings.manager.offline:
            self._load_backend()

            # Point the BEST Antenna
            if settings.instrument.enable_pointing:
                self._instrument = BEST2.Instance()

                try:
                    self._instrument.move_to_declination(observation.declination)
                except BEST2PointingException:
                    log.warning('Could not point the BEST2 Antenna to DEC: {:0.2f}'.format(observation.declination))
            else:
                log.info('BEST-II pointing is disabled as specified in configuration')

        observation.run(observation)

    def stop(self):
        """
        Stop all the BIRALES system sub-modules

        :return:
        """

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




    def start_scheduler(self, schedule_file_path, file_format):
        """
        Start the scheduler

        :param schedule_file_path:
        :param file_format:
        :return:
        """

        try:
            # The Scheduler responsible for the scheduling of observations
            self._scheduler = ObservationsScheduler()

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

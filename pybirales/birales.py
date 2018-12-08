import logging as log
import os
import signal
import sys

from pybirales import settings
from pybirales.listeners.listeners import NotificationsListener
from pybirales.services.scheduler.exceptions import SchedulerException, NoObservationsQueuedException
from pybirales.services.scheduler.scheduler import ObservationsScheduler


class BiralesFacade:
    def __init__(self, configuration):
        # The configuration associated with this facade Instance
        self.configuration = configuration

        # Load the system configuration upon initialisation
        self.configuration.load()

        # Initialise and start the listeners / subscribers of the BIRALES application
        self._listeners = []
        if settings.observation.notifications:
            self._listeners = []

            for l in self._listeners:
                l.start()

        # The BIRALES scheduler
        self._scheduler = ObservationsScheduler()

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """

        :param signum:
        :param frame:
        :return:
        """

        log.info("Ctrl-C detected by process %s, stopping birales system", os.getpid())

        self.stop()

        sys.exit(0)

    def stop(self):
        """
        Stop all the BIRALES system sub-modules

        :return:
        """

        # Stop the Observations scheduler
        if self._scheduler is not None:
            log.debug('Stopping the Scheduler instance')
            self._scheduler.stop()

        # Stop the listener threads
        if self._listeners is not None:
            log.debug('Stopping the listeners')
            for l in self._listeners:
                l.stop()

            log.info('Listeners stopped')

    def start_scheduler(self, schedule_file_path, file_format):
        """
        Start the scheduler

        :param schedule_file_path:
        :param file_format:
        :return:
        """

        try:
            # The Scheduler responsible for the scheduling of observations
            self._scheduler.start(schedule_file_path, file_format)
        except KeyboardInterrupt:
            log.info('Ctrl-C received. Terminating the scheduler process.ws')
        except NoObservationsQueuedException:
            log.info('Could not run Scheduler since no valid observations could be scheduled.')
        except SchedulerException:
            log.exception('A fatal Scheduler error has occurred. Terminating the scheduler process.')
        else:
            log.info('Scheduler finished successfully.')
        finally:
            self.stop()

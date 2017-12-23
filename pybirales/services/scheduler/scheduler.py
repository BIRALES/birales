import datetime
import dateutil.parser
import humanize
import json

import pytz
import sched
import sys
import threading
import time
import logging as log

from pybirales.birales import BiralesFacade, BiralesConfig
from pybirales.services.scheduler.exceptions import NoObservationsQueuedException, ObservationScheduledInPastException, \
    ObservationsConflictException
from pybirales.services.scheduler.observation import ScheduledObservation, ScheduledCalibrationObservation
from pybirales.services.scheduler.queue import ObservationsQueue
from pybirales.services.scheduler.monitoring import monitor_worker


def run_observation(observation):
    """
    Run the pipeline using the provided observation settings

    :param observation: A ScheduledObservation object
    :return:
    """

    # Load the BIRALES configuration from file
    config = BiralesConfig(observation.config_file, observation.parameters)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Build the pipeline manager
    manager = bf.build_pipeline(observation.pipeline_builder)

    # Start observation
    if isinstance(observation, ScheduledObservation):
        bf.start_observation(pipeline_manager=manager)

    # Calibrate the Instrument
    if isinstance(observation, ScheduledCalibrationObservation):
        bf.calibrate(correlator_pipeline_manager=manager)


class Scheduler:
    # By default, all observation have a delayed start by this amount
    DEFAULT_WAIT_SECONDS = 5

    def __init__(self):
        """
        Initialise the Scheduler class

        """

        # The sched scheduler instance
        self._scheduler = sched.scheduler(time.time, time.sleep)

        # A queue of observation objects
        self._observations_queue = ObservationsQueue()

        # The maximum amount of time BIRALES will run before re-calibrating (specified in hours)
        self._max_uncalibrated_threshold = 24

        # Estimated time taken for a calibration observation (specified in minutes)
        self._calibration_time = 30

        # Monitoring thread that will output the status of the scheduler at specific intervals
        self._monitor_thread = threading.Thread(target=monitor_worker, args=(self._scheduler,))

    def load_from_json_schedule(self, schedule_file_path):
        """
        Schedule a series of observations described in a JSON input file

        :param schedule_file_path: The file path to the schedule in JSON format
        :return: None
        """

        # Open JSON file and convert it to a dictionary that can be iterated on
        with open(schedule_file_path) as json_data:
            scheduled_observations = json.load(json_data)

        # Create ScheduledObservation objects from the input file
        observations = [ScheduledObservation(name=obs_name, params=obs) for obs_name, obs in
                        scheduled_observations.iteritems()]

        # Schedule the observations
        self._add_observations(observations)

    @staticmethod
    def load_from_tdm_schedule(schedule_file_path):
        """
        Schedule a series of observations described in a TDM input file

        :param schedule_file_path: The file path to the schedule in TDM format
        :return: None
        """

        # todo - Implement the functionality (follow JSON variant's implementation)
        # Open TDM file and convert it to a dictionary that can be iterated on
        #

        # Create ScheduledObservation objects from the input file
        #

        # Schedule the observations
        # self._add_observations(observations)

    def start(self):
        """
        Start the scheduler

        :return:
        """

        # Check if there are scheduled observations
        if not self._observations_queue:
            raise NoObservationsQueuedException()

        # Schedule the observations
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        for observation in self._observations_queue:
            wait_seconds = (observation.start_time_padded - now).total_seconds()
            if wait_seconds < 1:
                self._scheduler.enter(delay=wait_seconds, priority=0, action=run_observation, argument=(observation,))

        # Start the monitoring thread
        self._monitor_thread.start()

        try:
            # Start the scheduler
            self._scheduler.run()
        except KeyboardInterrupt:
            log.info('Ctrl-C received. Terminating the scheduler process.')
            sys.exit()

    def _add_observations(self, observations):
        """
        Schedule a list of observations

        :param observations: A list of observation objects to be scheduled
        :type observations: list of ScheduledObservation objects
        :return:
        """

        # Schedule the observations
        for observation in observations:
            self._add_observation(observation)

        # Show that start time of all the observations to the user
        log.info('{} observations queued.'.format(len(self._observations_queue)))
        for observation in self._observations_queue:
            log.info(observation.start_message())

    def _is_calibration_required(self):
        pass

    def _add_calibration_observation(self):
        """
        Schedule a calibration observation

        :return:
        """

        return None

    def _get_earliest_calibration(self):
        """
        Determine the earliest scheduled calibration routine

        :return:
        """

        pass



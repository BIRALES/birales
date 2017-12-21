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
from pybirales.services.scheduler.observation import ScheduledObservation


def monitor_worker(scheduler):
    """
    Start the monitoring thread

    :return:
    """

    while not scheduler.empty():
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        for observation in scheduler.queue:
            obs_settings = observation.argument[0]
            start_time = dateutil.parser.parse(obs_settings['config_parameters']['start_time'])
            time_remaining = humanize.naturaltime(now - start_time)

            log.info('The %s for the "%s" observation is scheduled to start in %s',
                     obs_settings['pipeline'],
                     obs_settings['name'],
                     time_remaining)

        # Do not show the output again for the next N seconds
        time.sleep(60)


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
    bf.start_observation(pipeline_manager=manager)


def run_calibration(observation):
    """
    Run calibration routine

    :param observation:
    :return:
    """

    # Load the BIRALES configuration from file
    config = BiralesConfig(observation.config_file, observation.parameters)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Build the Pipeline Manager using the Correlator Pipeline Manager Builder
    manager = bf.build_pipeline(observation.pipeline_builder)

    # Calibrate the Instrument
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
        self._observation_queue = []

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
        if not self._observation_queue:
            raise NoObservationsQueuedException()

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
        log.info('{} observations queued.'.format(len(self._observation_queue)))
        for observation in self._observation_queue:
            log.info(observation.start_message())

    def _add_observation(self, observation):
        """
        Schedule an observation

        :param observation: The observation to be scheduled
        :type observation: ScheduledObservation
        :return:
        """

        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

        try:
            self._observation_conflict(observation)

            wait_seconds = (observation.start_time_padded - now).total_seconds()

            if wait_seconds < 1:
                raise ObservationScheduledInPastException(observation.parameters, observation.start_time_padded)

        except ObservationsConflictException:
            log.error("Observation '{}', could not be scheduled".format(observation.name))
        except ObservationScheduledInPastException:
            log.error("Observation '{}' could not be scheduled".format(observation.name))
        else:
            # Add the observation to the queue
            self._observation_queue.append(observation)

            # Schedule the observation
            self._scheduler.enter(delay=wait_seconds,
                                  priority=0,
                                  action=run_observation,
                                  argument=(observation,))

            log.info("Observation '{}', queued successfully".format(observation.name))

    def _is_calibration_required(self):
        pass

    def _add_calibration_observation(self):
        """
        Schedule a calibration observation

        :return:
        """
        pass

    def _get_earliest_calibration(self):
        """
        Determine the earliest scheduled calibration routine

        :return:
        """

        pass

    def _observation_conflict(self, observation):
        """
        Check that the passed on observation does not conflict with the queued observations

        :param observation: The observation to be scheduled
        :type observation: ScheduledObservation
        :raises ObservationsConflictException: The observation conflicts with the queued observations
        :return: None
        """

        start_time = observation.start_time_padded
        end_time = observation.end_time_padded

        for scheduled_observation in self._observation_queue:
            so_start = scheduled_observation.start_time_padded
            so_end = scheduled_observation.end_time_padded

            # Check for time ranges where duration is not defined:
            if not (so_end and end_time):
                # If there is an observation that has no end and starts before this observation
                if not so_end and so_start < start_time:
                    raise ObservationsConflictException(observation, scheduled_observation)

                # If this observation starts before and does not end
                if not end_time and start_time < so_start:
                    raise ObservationsConflictException(observation, scheduled_observation)

            # If this observation overlaps another time range
            elif (start_time <= so_end) and (so_start <= end_time):
                raise ObservationsConflictException(observation, scheduled_observation)
        else:
            # No overlap detected
            log.info('No overlap detected between scheduled observations')

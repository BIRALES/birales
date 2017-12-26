import datetime
import json
import logging as log
import sched
import sys
import threading
import time
from operator import attrgetter
import dateutil.parser

import pytz
from pybirales.birales import BiralesFacade, BiralesConfig
from pybirales.services.scheduler.exceptions import NoObservationsQueuedException, IncorrectScheduleFormat
from pybirales.services.scheduler.monitoring import monitor_worker
from pybirales.services.scheduler.observation import ScheduledObservation, ScheduledCalibrationObservation
from pybirales.services.scheduler.schedule import Schedule
from pybirales.pipeline.pipeline import get_builder_by_id


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
    manager = bf.build_pipeline(pipeline_builder=get_builder_by_id(observation.pipeline_name))

    # Start observation
    if isinstance(observation, ScheduledObservation):
        bf.start_observation(pipeline_manager=manager)

    # Calibrate the Instrument
    if isinstance(observation, ScheduledCalibrationObservation):
        bf.calibrate(correlator_pipeline_manager=manager)


class ObservationsScheduler:
    # By default, all observation have a delayed start by this amount
    DEFAULT_WAIT_SECONDS = 5

    def __init__(self):
        """
        Initialise the Scheduler class

        """

        # The sched scheduler instance
        self._scheduler = sched.scheduler(time.time, time.sleep)

        # The maximum amount of time BIRALES will run before re-calibrating (specified in hours)
        self._max_uncalibrated_threshold = datetime.timedelta(hours=24)

        # Estimated time taken for a calibration observation (specified in minutes)
        self._calibration_time = datetime.timedelta(minutes=30)

        # A queue of observation objects
        self._schedule = Schedule(time_to_calibrate=self._calibration_time,
                                  recalibration_time=self._max_uncalibrated_threshold)

        # Monitoring thread that will output the status of the scheduler at specific intervals
        self._monitor_thread = threading.Thread(target=monitor_worker, args=(self._scheduler,))

    def load_from_file(self, schedule_file_path, file_format='json'):
        """
        Schedule a series of observations described in a JSON input file

        :param schedule_file_path: The file path to the schedule
        :param file_format: The format of the schedule
        :return: None
        """

        obs = []
        if file_format == 'json':
            # Open JSON file and convert it to a dictionary that can be iterated on
            with open(schedule_file_path) as json_data:
                obs = json.load(json_data)

        if obs:
            # Schedule the observations
            self._add_observations(obs)
        else:
            raise IncorrectScheduleFormat(schedule_file_path)

    def start(self):
        """
        Start the scheduler

        :return:
        """

        # Check if there are scheduled observations
        if not self._schedule.is_empty():
            raise NoObservationsQueuedException()

        # Schedule the observations
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        for observation in self._schedule:
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

    def _add_observations(self, scheduled_observations):
        """
        Schedule a list of observations

        :param scheduled_observations: A list of observation objects to be scheduled
        :type scheduled_observations: list of ScheduledObservation objects
        :return:
        """

        # Create ScheduledObservation objects from the input file
        observations = []
        for obs_name, obs in scheduled_observations.iteritems():
            # todo - parameters should be handled better than this
            try:
                so = ScheduledObservation(name=obs_name,
                                          config_file=obs['config_file'],
                                          pipeline_name=obs['pipeline'],
                                          dec=obs['config_parameters']['beamformer']['reference_declination'],
                                          start_time=dateutil.parser.parse(obs['config_parameters']['start_time']),
                                          duration=obs['config_parameters']['duration'])
                observations.append(so)
            except KeyError:
                log.exception('An error occurred. Some parameters are missing.')

        # It is assumed that the list is sorted by date
        sorted_observations = sorted(observations, key=attrgetter('start_time_padded'))
        # Schedule the observations
        for observation in sorted_observations:
            self._schedule.add_observation(observation)

        # Show that start time of all the observations to the user
        log.info('{} observations and {} calibration observations queued.'.format(
            self._schedule.n_observations, self._schedule.n_calibrations))

    def _output_start_messages(self):
        for observation in self._schedule:
            log.info(observation.start_message())

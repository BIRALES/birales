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
from pybirales.pipeline.pipeline import get_builder_by_id
from pybirales.services.scheduler.exceptions import NoObservationsQueuedException, IncorrectScheduleFormat, \
    InvalidObservationException
from pybirales.services.scheduler.monitoring import monitor_worker
from pybirales.services.scheduler.observation import ScheduledObservation, ScheduledCalibrationObservation
from pybirales.services.scheduler.schedule import Schedule


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
        self._monitor_thread = threading.Thread(target=monitor_worker, args=(self._scheduler,), name='Monitoring')

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
                try:
                    obs = json.load(json_data)
                except ValueError:
                    log.exception('JSON file at {} is malformed.'.format(schedule_file_path))
                    raise IncorrectScheduleFormat(schedule_file_path)

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
        if self._schedule.is_empty():
            raise NoObservationsQueuedException()

        # Schedule the observations
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        for observation in self._schedule:
            wait_seconds = (observation.start_time_padded - now).total_seconds()
            if wait_seconds > 1:
                self._scheduler.enter(delay=wait_seconds, priority=0, action=run_observation, argument=(observation,))

        # Start the monitoring thread
        self._monitor_thread.start()

        # Start the scheduler
        self._scheduler.run()

    def stop(self):
        log.info('Cancelling {} observations from schedule'.format(len(self._scheduler.queue)))
        for event in self._scheduler.queue:
            self._scheduler.cancel(event)
            log.debug('The `{}` observation was cancelled'.format(event.argument[0].name))

        if self._scheduler.empty():
            log.info('Scheduler was cleared from all events. Please wait for the monitoring thread to terminate.')

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
            try:
                so = ScheduledObservation(name=obs_name,
                                          config_file=obs['config_file'],
                                          pipeline_name=obs['pipeline'],
                                          params=obs['config_parameters'])
                observations.append(so)
            except KeyError:
                log.exception('An incorrect parameter was specified in observation `{}`'.format(obs_name))
                log.warning('Observation `{}` was not added to the schedule'.format(obs_name))
                continue
            except InvalidObservationException:
                log.exception('An incorrect parameter was specified in observation `{}`'.format(obs_name))
                log.warning('Observation `{}` was not added to the schedule'.format(obs_name))
                continue

        # It is assumed that the list is sorted by date
        sorted_observations = sorted(observations, key=attrgetter('start_time_padded'))
        # Schedule the observations
        for observation in sorted_observations:
            try:
                self._schedule.add_observation(observation)
            except InvalidObservationException:
                log.warning('Observation `{}` was not added to the schedule'.format(observation.name))

        # Show that start time of all the observations to the user
        log.info('{} observations and {} calibration observations queued.'.format(
            self._schedule.n_observations, self._schedule.n_calibrations))

    def _output_start_messages(self):
        for observation in self._schedule:
            log.info(observation.start_message())

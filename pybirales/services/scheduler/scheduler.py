import datetime
import json
import logging as log
import sched
import threading
import time
from operator import attrgetter

from pybirales.events.events import ObservationScheduledCancelledEvent
from pybirales.events.publisher import EventsPublisher
from pybirales.repository.message_broker import RedisManager
from pybirales.repository.models import Observation as ObservationModel
from pybirales.services.scheduler.exceptions import IncorrectScheduleFormat, \
    InvalidObservationException
from pybirales.services.scheduler.monitoring import monitor_worker, obs_listener_worker
from pybirales.services.scheduler.observation import ScheduledObservation, ScheduledCalibrationObservation
from pybirales.services.scheduler.schedule import Schedule


class ObservationsScheduler:
    # By default, all observation have a delayed start by this amount
    DEFAULT_WAIT_SECONDS = 5
    OBSERVATIONS_CHL = 'birales_scheduled_obs'

    def __init__(self):
        """
        Initialise the Scheduler class

        """

        # The sched scheduler instance
        self._scheduler = sched.scheduler(time.time, time.sleep)

        # A queue of observation objects
        self._schedule = Schedule()

        # Event published of the application
        self._publisher = EventsPublisher.Instance()

        # Monitoring thread that will output the status of the scheduler at specific intervals
        self._monitor_thread = threading.Thread(target=monitor_worker, args=(self._scheduler,), name='Monitoring')

        # Create an observations thread which listens for new observations (through pub-sub)
        self._obs_thread = threading.Thread(target=obs_listener_worker, args=(self._scheduler,), name='Obs. Listener')

        # The redis instance
        self._redis = RedisManager.Instance().redis

        # The PubSub interface of the redis instance
        self._pubsub = self._redis.pubsub()

        # Subscribe to the observations channel
        self._pubsub.subscribe(self.OBSERVATIONS_CHL)

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

        # # Check if there are scheduled observations
        # if self._schedule.is_empty():
        #     raise NoObservationsQueuedException()

        # Start the monitoring thread
        self._monitor_thread.start()

        # Start the scheduler
        self._scheduler.run()

        # Queue existing observations that were stored in the database
        self._load_observations()

        # Listen in for new observations (blocking)
        self._listen()

    def _load_observations(self):
        """
        Query the database for observations that were already scheduled

        :return:
        """
        now = datetime.datetime.utcnow()

        # Get observations from the database
        observations = ObservationModel.objects(date_time_start__gte=now)

        log.info('Loading %s observations from database', len(observations))

        # Schedule the observations
        for obs in observations:
            try:
                if obs.type == 'observation':
                    self._schedule.add_observation(ScheduledObservation(name=obs.name,
                                                                        pipeline_name=obs.pipeline_name,
                                                                        config_file=obs.config_filepath,
                                                                        params=obs.settings))
                elif obs.type == 'calibration':
                    self._schedule.add_observation(ScheduledCalibrationObservation(name=obs.name,
                                                                                   config_file=obs.config_filepath,
                                                                                   params=obs.settings))

            except InvalidObservationException:
                log.warning('Observation `{}` was not added to the schedule'.format(obs.name))

        # Queue the observation to the sched instance
        self._queue_observations()

    def _listen(self):
        log.info('Scheduler listening on `{}` for new observations'.format(self.OBSERVATIONS_CHL))
        for message in self._pubsub.listen():
            if message['data'] == 'KILL':
                log.info('Scheduled observations listener un-subscribed from {}'.format(self.OBSERVATIONS_CHL))
                break
            else:
                if message['type'] == 'message':
                    log.info("New observation received by scheduler: {}".format(message['data']))
                    self._add_observations(json.loads(message['data']))

    def stop(self):
        """
        Stop the scheduler gracefully
        :return:
        """

        log.info('Cancelling {} observations from schedule'.format(len(self._scheduler.queue)))

        # Stop listening for new observations
        self._redis.publish(self.OBSERVATIONS_CHL, 'KILL')

        # Clear the schedule from the observations
        for event in self._scheduler.queue:
            self._scheduler.cancel(event)
            self._publisher.publish(ObservationScheduledCancelledEvent(observation=event.argument[0]))

        if self._scheduler.empty():
            log.info('Scheduler was cleared from all events. Please wait for the monitoring thread to terminate.')

        # Wait for all the notifications to be sent
        time.sleep(2)

    def _add_observations(self, scheduled_observations):
        """
        Schedule a list of observations

        :param scheduled_observations: A list of observation objects to be scheduled
        :type scheduled_observations: list of ScheduledObservation objects
        :return:
        """

        # Create the ScheduledObservation Objects
        observations = self._create_obs(scheduled_observations)

        # Schedule the observations
        for observation in observations:
            try:
                self._schedule.add_observation(observation)
            except InvalidObservationException:
                log.warning('Observation `{}` was not added to the schedule'.format(observation.name))

        # Queue the observation to the sched instance
        self._queue_observations()

    @staticmethod
    def _create_obs(scheduled_observations):
        """
        Create ScheduledObservation objects

        :param scheduled_observations:
        :return:
        """

        observations = []
        for obs_name, obs in scheduled_observations.iteritems():
            try:
                if obs['type'] == 'observation':
                    so = ScheduledObservation(name=obs_name,
                                              pipeline_name=obs['pipeline'],
                                              config_file=obs['config_file'],
                                              params=obs['config_parameters'])
                elif obs['type'] == 'calibration':
                    so = ScheduledCalibrationObservation(name=obs_name,
                                                         config_file=obs['config_file'],
                                                         params=obs['config_parameters'])
                else:
                    raise InvalidObservationException

                observations.append(so)
            except KeyError:
                log.exception('Incorrect/missing parameter in observation `{}`'.format(obs_name))
                log.warning('Observation `{}` was not added to the schedule'.format(obs_name))
                continue
            except InvalidObservationException:
                log.exception('An incorrect parameter was specified in observation `{}`'.format(obs_name))
                log.warning('Observation `{}` was not added to the schedule'.format(obs_name))
                continue

        # Return a sorted list of observations by their start time
        return sorted(observations, key=attrgetter('start_time_padded'))

    def _queue_observations(self):
        """
        Add the observations to the sched instance
        :return:
        """

        for observation in self._schedule:
            # Only add observations that were not added already in sched
            if observation.event not in self._scheduler.queue:
                if observation.wait_time > 1:
                    # Schedule this observation, using sched
                    event = self._scheduler.enter(delay=observation.wait_time, priority=0,
                                                  action=observation.run,
                                                  argument=(observation,))

                    # Associate the scheduled event with this observation
                    observation.event = event
                else:
                    log.warning('Observation `{}` was not added to the schedule'.format(observation.name))

        log.info('{} observations and {} calibration observations in queue.'.format(self._schedule.n_observations,
                                                                                    self._schedule.n_calibrations))

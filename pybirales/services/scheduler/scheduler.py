import datetime
import json
import logging as log
import signal
import threading
import time

import humanize
import pytz

from pybirales.events.events import BIRALESSchedulerReloadedEvent
from pybirales.events.publisher import publish
from pybirales.repository.message_broker import broker
from pybirales.repository.models import Observation as ObservationModel
from pybirales.services.scheduler.exceptions import IncorrectScheduleFormat, InvalidObservationException
from pybirales.services.scheduler.monitoring import obs_listener_worker
from pybirales.services.scheduler.observation import ScheduledObservation, ScheduledCalibrationObservation
from pybirales.services.scheduler.schedule import Schedule


class ObservationsScheduler:
    # By default, all observation have a delayed start by this amount
    DEFAULT_WAIT_SECONDS = 5
    OBSERVATIONS_CHL = 'birales_scheduled_obs'
    BIRALES_STATUS_CHL = 'birales_system_status'
    POLL_FREQ = datetime.timedelta(seconds=5)

    def __init__(self):
        """
        Initialise the Scheduler class

        """

        # A queue of observation objects
        self._schedule = Schedule()

        self._stop_event = threading.Event()

        self._reload_event = threading.Event()

        # Create an observations thread which listens for new observations (through pub-sub)
        self._obs_thread = threading.Thread(target=obs_listener_worker, args=(self,),
                                            name='Obs. Listener')

    def load_from_file(self, schedule_file_path, file_format='json'):
        """
        Schedule a series of observations described in a JSON input file

        :param schedule_file_path: The file path to the schedule
        :param file_format: The format of the schedule
        :return: None
        """

        observations = []
        if file_format == 'json':
            # Open JSON file and convert it to a dictionary that can be iterated on
            with open(schedule_file_path) as json_data:
                try:
                    observations = json.load(json_data)
                except ValueError:
                    log.exception('JSON file at {} is malformed.'.format(schedule_file_path))
                    raise IncorrectScheduleFormat(schedule_file_path)

        if observations:
            scheduled_observation = self._add_observations(observations)
        else:
            raise IncorrectScheduleFormat(schedule_file_path)

    def start(self, schedule_file_path=None, file_format=None):
        """
        Start the scheduler

        :return:
        """

        # Restore existing observations that were stored in the database
        self._restore_observations()

        # Add new observations that were specified in schedule file
        if schedule_file_path:
            self.load_from_file(schedule_file_path, file_format)

        # Start the monitoring thread
        self._obs_thread.start()

        # Start the scheduler
        self.run()

        # Wait for a keyboard interrupt to exit the process
        signal.pause()

    def _restore_observations(self):
        """
        Restore observations from the database

        :return:
        """

        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

        scheduled_observations = ObservationModel.objects(date_time_start__gte=now)

        if scheduled_observations:
            log.info('Restoring %s observations from database', len(scheduled_observations))
            self._add_observations(scheduled_observations)
        else:
            log.info('No observations found in database.')

    def reload(self):
        """

        :return:
        """
        log.debug('Reloading scheduler')

        # Empty the schedule by creating one afresh
        self._schedule = Schedule()

        # Restore the observation from the database
        self._restore_observations()

        publish(BIRALESSchedulerReloadedEvent())

        log.info('Scheduler reloaded')

    def run(self):
        """

        :return:
        """

        log.info('BIRALES Scheduler observation runner started')
        counter = 0
        while not (self._stop_event.is_set() or self._reload_event.is_set()):
            next_observation = self.schedule.next_observation
            now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

            if counter % 10 == 0:
                self._monitoring_message(now, next_observation)

            if next_observation:

                if (now - next_observation.start_time_padded).total_seconds() > 2:
                    next_observation.manager.run(next_observation)

            counter += 1
            time.sleep(1)

        log.info('BIRALES Scheduler observation runner stopped')

        # Clear the reload flag such that the scheduler can be reloaded again
        self._reload_event.clear()

    def _monitoring_message(self, now, next_observation):
        if len(self._schedule) < 1:
            log.info('No queued observations.')
        else:
            log.info('%s observations and %s calibration observations in queue. Next observation: %s',
                     self._schedule.n_observations,
                     self._schedule.n_calibrations, next_observation.name)

            for event in self._schedule:
                delta = now - event.start_time_padded
                log.info('The %s for the `%s` observation is scheduled to start in %s and will run for %s',
                         event.pipeline_name,
                         event.name,
                         humanize.naturaldelta(delta),
                         humanize.naturaldelta(event.duration))

        broker.publish(self.BIRALES_STATUS_CHL, json.dumps({
            'scheduler': {
                'status': 'running',
                'obs_thread_status': self._obs_thread.isAlive(),
                'timestamp': now.isoformat('T'),
                'next_update': (now + self.POLL_FREQ).isoformat('T'),
                'dt': self.POLL_FREQ.seconds
            }
        }))

    def stop(self):
        """
        Stop the scheduler gracefully
        :return:
        """

        log.info('Cancelling {} observations from schedule'.format(len(self.schedule)))

        # Stop listening for new observations
        broker.publish(self.OBSERVATIONS_CHL, 'KILL')

        # stop monitoring thread
        self._stop_event.set()

        # for t in threading.enumerate():
        #     print t.getName()

    def _add_observations(self, scheduled_observations):
        """
        Schedule a list of observations

        :param scheduled_observations: A list of observation objects to be scheduled
        :type scheduled_observations: list of ScheduledObservation objects
        :return:
        """

        # List of observations that were successfully added to the schedule
        scheduled_obs = []

        # Schedule the observations
        for obs in scheduled_observations:
            try:
                if 'name' not in obs:
                    raise InvalidObservationException('Missing name parameter for observation')

                # Create the ScheduledObservation Objects
                observation = self.create_obs(obs)

                # Add the scheduled objects to the queue
                self._schedule.add_observation(observation)
            except InvalidObservationException:
                log.warning('Observation %s was not added to the schedule', obs['name'])
            else:
                scheduled_obs.append(observation)

        return scheduled_obs

    @property
    def schedule(self):
        return self._schedule

    def clear(self):
        del self._schedule
        self._schedule = Schedule()

    @staticmethod
    def create_obs(obs):
        """
        Create ScheduledObservation objects

        :param scheduled_observations:
        :return:
        """

        try:
            _id = None
            if 'id' in obs:
                _id = obs['id']
            if obs['type'] == 'observation':
                so = ScheduledObservation(name=obs['name'],
                                          pipeline_name=obs['pipeline'],
                                          config_file=obs['config_file'],
                                          config_parameters=obs['config_parameters'],
                                          model_id=_id)
            elif obs['type'] == 'calibration':
                so = ScheduledCalibrationObservation(name=obs['name'],
                                                     pipeline_name=None,
                                                     config_file=obs['config_file'],
                                                     config_parameters=obs['config_parameters'],
                                                     model_id=_id)
            else:
                raise InvalidObservationException('Observation type is not valid.')

        except KeyError:
            log.exception('Incorrect/missing parameter in observation %s', obs)
            log.warning('Observation %s was not added to the schedule', obs)

            raise InvalidObservationException('Incorrect/missing parameter in observation')
        except InvalidObservationException:
            log.exception('An incorrect parameter was specified in observation %s', obs)
            log.warning('Observation %s was not added to the schedule', obs)
        else:
            return so

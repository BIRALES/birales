import datetime
import json
import logging as log
import os
import signal
import threading
import time

import humanize
import pytz

from pybirales.base.observation_manager import ObservationManager, CalibrationObservationManager
from pybirales.events.events import InvalidObservationEvent
from pybirales.events.publisher import publish
from pybirales.repository.message_broker import broker
from pybirales.services.scheduler.exceptions import IncorrectScheduleFormat, InvalidObservationException
from pybirales.services.scheduler.monitoring import obs_listener_worker
from pybirales.services.scheduler.observation import ScheduledObservation, ScheduledCalibrationObservation
from pybirales.services.scheduler.schedule import Schedule
from pybirales.birales_config import BiralesConfig


class ObservationsScheduler:
    DEFAULT_WAIT_SECONDS = 5  # By default, all observation have a delayed start by this amount
    OBSERVATIONS_CHL = b'birales_scheduled_obs'
    BIRALES_STATUS_CHL = b'birales_system_status'
    POLL_FREQ = datetime.timedelta(seconds=5)
    MONITORING_FREQ = 10

    def __init__(self):
        """
        Initialise the Scheduler class

        """

        # A queue of observation objects
        self._schedule = Schedule()

        self._stop_event = threading.Event()

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

    def get_mock_obs(self):
        """

        :return:
        """

        log.info("injecting obs")
        time.sleep(1)

        root_dir = os.path.join(os.environ['HOME'], '.birales')
        config_parameters = {
            "beamformer": {
                "reference_declination": float(45.)
            },
            "observation": {
                "name": 'T2',
                'transmitter_frequency': 410.085
            },
            "start_time": '{:%Y-%m-%d %H:%M:%S}Z'.format(datetime.datetime.utcnow() + datetime.timedelta(seconds=10)),
            "duration": 50
        }

        data = json.dumps({
            "name": 'T2',
            "type": 'observation',
            "pipeline": 'detection_pipeline',
            "config_file": [
                os.path.join(root_dir, "configuration/birales.ini"),
                os.path.join(root_dir, "configuration/detection.ini"),
            ],
            "config_parameters": config_parameters
        })

        broker.publish('birales_scheduled_obs', data)

    def start(self, schedule_file_path=None, file_format=None):
        """
        Start the scheduler

        :return:
        """

        # Add new observations that were specified in schedule file
        if schedule_file_path:
            self.load_from_file(schedule_file_path, file_format)

        # Start the monitoring thread
        self._obs_thread.start()

        # Start the scheduler
        self.run()

        # Wait for a keyboard interrupt to exit the process
        signal.pause()

    @property
    def schedule(self):
        return self._schedule

    def run(self):
        """

        :return:
        """

        log.info('BIRALES Scheduler observation runner started')
        counter = 0
        processed_observations = []
        om = ObservationManager()
        com = CalibrationObservationManager()

        while not self._stop_event.is_set():
            now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
            pending_observations = self.schedule.pending_observations()

            if pending_observations:
                next_observation = pending_observations[0]

                if next_observation.should_start and next_observation.id not in processed_observations:
                    processed_observations.append(next_observation.id)

                    if isinstance(next_observation, ScheduledCalibrationObservation):
                        # Run the calibration observation using the observation manager
                        com.run(next_observation)
                    else:
                        # Run the observation using the observation manager
                        om.run(next_observation)

            if counter % self.MONITORING_FREQ == 0:
                self._monitoring_message(now, pending_observations)
                # all_objects = muppy.get_objects()
                # sum1 = summary.summarize(all_objects)
                # summary.print_(sum1)

                # print mem_top()
                # tr.print_diff()

                # print h.heap()

            counter += 1
            time.sleep(1)

        log.info('BIRALES Scheduler observation runner stopped')

    def _get_active_threads(self):
        """
        Return the threads that are still active

        :return:
        """
        main_thread = threading.current_thread()
        active_threads = []
        for t in threading.enumerate():
            if t is main_thread:
                continue
            active_threads.append(t.getName())
        return ",".join(map(str, active_threads))

    def _monitoring_message(self, now, pending_observations):
        if len(pending_observations) < 1:
            log.info('No queued observations.')

            if a_threads := [t.getName() for t in threading.enumerate() if t.is_alive()]:
                log.warning('Running threads: %s', ', '.join(a_threads))
        else:
            log.info('There are %s observation queued. Next observation: %s', len(pending_observations),
                     pending_observations[0].name)
            for obs in pending_observations:
                delta = now - obs.start_time
                log.info('The %s for the `%s` observation is scheduled to start in %s and will run for %s',
                         obs.pipeline_name,
                         obs.name,
                         humanize.naturaldelta(delta),
                         humanize.naturaldelta(obs.duration))

        broker.publish(self.BIRALES_STATUS_CHL, json.dumps({
            'scheduler': {
                'status': 'running',
                'obs_thread_status': self._obs_thread.is_alive(),
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

        log.info('Cancelling {} observations from schedule'.format(len(self.schedule.observations)))

        # Stop listening for new observations
        broker.publish(self.OBSERVATIONS_CHL, 'KILL')

        # Stop the Pipeline control thread
        broker.publish('birales_pipeline_control', 'KILL')

        # Stop the slack notifications thread
        broker.publish('slack_notifications', 'KILL')

        # stop monitoring thread
        self._stop_event.set()

        time.sleep(1)
        for t in threading.enumerate():
            log.debug('{} is still running'.format(t.getName()))

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
                self._schedule.add(observation)
            except InvalidObservationException:
                log.warning('Observation %s was not added to the schedule', obs['name'])
                publish(InvalidObservationEvent(obs, 'Could not add observation'))
            else:
                scheduled_obs.append(observation)

        return scheduled_obs

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


if __name__ == '__main__':
    CONFIG_ROOT = os.path.join(os.environ['HOME'], '.birales/configuration/')
    DEFAULT_CONFIG = CONFIG_ROOT + 'birales.ini'
    DETECTION_CONFIG = CONFIG_ROOT + 'detection.ini'
    date_str = '2021-10-07'
    date_str += ' '
    OBSERVATIONS = [['NORAD40699', date_str + '08:59:00', date_str + '09:03:00', 0.07],
                    ['NORAD37782', date_str + '09:04:00', date_str + '09:09:00', 60.20],
                    ['NORAD38771', date_str + '09:12:00', date_str + '09:18:00', 13.07],
                    ['NORAD22626', date_str + '09:20:00', date_str + '09:32:00', 63.10],
                    ['NORAD26536', date_str + '09:33:00', date_str + '09:37:00', 58.74],
                    ['NORAD5560', date_str + '09:39:00', date_str + '09:44:00', 39.21],
                    ['NORAD20443', date_str + '09:45:00', date_str + '09:51:00', 69.91],
                    ['NORAD27421', date_str + '09:52:00', date_str + '09:58:00', 32.88],
                    ['NORAD37387', date_str + '10:01:00', date_str + '10:14:00', 2.46],
                    ['NORAD10967', date_str + '10:15:00', date_str + '10:28:00', 19.00],
                    ['NORAD28651', date_str + '10:29:00', date_str + '10:32:00', 53.47],
                    ['NORAD12443', date_str + '10:33:00', date_str + '10:37:00', 75.43],
                    ['NORAD11288', date_str + '10:39:00', date_str + '10:46:00', 44.39],
                    ['NORAD3530', date_str + '10:47:00', date_str + '11:00:00', 22.84]]

    config = BiralesConfig([DEFAULT_CONFIG], {})
    config.load()
    scheduler = ObservationsScheduler()
    for obs in OBSERVATIONS:
        name, date_start_str, date_end_str, declination = obs
        pipeline = 'detection_pipeline'
        config_file = [DEFAULT_CONFIG, DETECTION_CONFIG]

        date_start = datetime.datetime.strptime(date_start_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.utc)
        date_end = datetime.datetime.strptime(date_end_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.utc)

        duration = (date_end - date_start).total_seconds()

        config_parameters = {
            "beamformer": {
                "reference_declination": declination
            },
            "observation": {
                "name": name,
                "target_name": name
            },
            "start_time": date_start,
            "duration": duration
        }

        so = ScheduledObservation(name=name,
                                  pipeline_name=pipeline,
                                  config_file=config_file,
                                  config_parameters=config_parameters)
        print(config_parameters)

        scheduler._schedule.add(so)

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
    date_str = '2021-11-04'
    date_str += ' '
    # OBSERVATIONS = [['NORAD40699', date_str + '08:59:00', date_str + '09:03:00', 0.07],
    #                 ['NORAD37782', date_str + '09:04:00', date_str + '09:09:00', 60.20],
    #                 ['NORAD38771', date_str + '09:12:00', date_str + '09:18:00', 13.07],
    #                 ['NORAD22626', date_str + '09:20:00', date_str + '09:32:00', 63.10],
    #                 ['NORAD26536', date_str + '09:33:00', date_str + '09:37:00', 58.74],
    #                 ['NORAD5560', date_str + '09:39:00', date_str + '09:44:00', 39.21],
    #                 ['NORAD20443', date_str + '09:45:00', date_str + '09:51:00', 69.91],
    #                 ['NORAD27421', date_str + '09:52:00', date_str + '09:58:00', 32.88],
    #                 ['NORAD37387', date_str + '10:01:00', date_str + '10:14:00', 2.46],
    #                 ['NORAD10967', date_str + '10:15:00', date_str + '10:28:00', 19.00],
    #                 ['NORAD28651', date_str + '10:29:00', date_str + '10:32:00', 53.47],
    #                 ['NORAD12443', date_str + '10:33:00', date_str + '10:37:00', 75.43],
    #                 ['NORAD11288', date_str + '10:39:00', date_str + '10:46:00', 44.39],
    #                 ['NORAD3530', date_str + '10:47:00', date_str + '11:00:00', 22.84]]

    # OBSERVATIONS = [['NORAD45389', date_str + '18:00:30', date_str + '18:03:30', 4.07],
    #                  ['NORAD46567', date_str + '18:06:50', date_str + '18:10:50', 49.73],
    #                  ['NORAD46589', date_str + '18:13:29', date_str + '18:16:29', 41.42],
    #                  ['NORAD44499', date_str + '18:18:37', date_str + '18:21:37', 50.77],
    #                  ['NORAD48097', date_str + '18:24:00', date_str + '18:27:00', -0.21],
    #                  ['NORAD29252', date_str + '18:29:36', date_str + '18:31:36', 47.82],
    #                  ['NORAD45099', date_str + '18:34:06', date_str + '18:37:06', 50.34],
    #                  ['NORAD48115', date_str + '18:38:46', date_str + '18:49:46', 21.01],
    #                  ['NORAD47832', date_str + '18:51:53', date_str + '18:54:53', 47.57],
    #                  ['NORAD45114', date_str + '18:56:55', date_str + '18:59:55', 1.65],
    #                  ['NORAD45377', date_str + '19:01:23', date_str + '19:03:23', 30.14],
    #                  ['NORAD42760', date_str + '19:06:54', date_str + '19:09:54', 16.27],
    #                  ['NORAD48014', date_str + '19:12:53', date_str + '19:15:53', 2.93],
    #                  ['NORAD42759', date_str + '19:19:26', date_str + '19:22:26', 15.57],
    #                  ['NORAD48134', date_str + '19:24:51', date_str + '19:26:51', 24.42],
    #                  ['NORAD42790', date_str + '19:29:43', date_str + '19:32:43', 72.34],
    #                  ['NORAD47828', date_str + '19:34:08', date_str + '19:36:08', 50.18],
    #                  ['NORAD45072', date_str + '19:39:40', date_str + '19:42:40', 30.56],
    #                  ['NORAD47876', date_str + '19:46:12', date_str + '19:49:12', 11.80],
    #                  ['NORAD48130', date_str + '19:51:31', date_str + '20:00:31', 13.19]]

    # OBSERVATIONS = [['NORAD41039', 	date_str + '19:00:35', 	date_str + '19:06:35', 9.45],
    #                 ['NORAD39450', 	date_str + '19:07:32', 	date_str + '19:09:32', 16.60],
    #                 ['NORAD17973', 	date_str + '19:10:01', 	date_str + '19:19:01', 33.75],
    #                 ['NORAD10962', 	date_str + '19:20:40', 	date_str + '19:36:40', 17.27],
    #                 ['NORAD23704', 	date_str + '19:38:51', 	date_str + '19:44:51', 36.15],
    #                 ['NORAD11870', 	date_str + '19:45:22', 	date_str + '19:51:22', 21.73],
    #                 ['NORAD23343', 	date_str + '19:53:03', 	date_str + '19:55:03', 6.09],
    #                 ['NORAD44835', 	date_str + '19:58:30', 	date_str + '20:01:30', 77.38],
    #                 ['NORAD17129', 	date_str + '20:03:29', 	date_str + '20:12:29', 37.63],
    #                 ['NORAD41728', 	date_str + '20:13:45', 	date_str + '20:20:45', 30.77],
    #                 ['NORAD28649', 	date_str + '20:22:07', 	date_str + '20:31:07', 53.28],
    #                 ['NORAD41335', 	date_str + '20:33:18', 	date_str + '20:35:18', 77.03],
    #                 ['NORAD36416', 	date_str + '20:37:29', 	date_str + '20:43:29', 49.98],
    #                 ['NORAD38347', 	date_str + '20:45:40', 	date_str + '20:47:40', 42.43],
    #                 ['NORAD33066', 	date_str + '20:49:38', 	date_str + '21:00:38', 69.67]]

    # OBSERVATIONS = [['NORAD40699', date_str + '20:00:41', date_str + '20:06:00', 11.34],
    #                 ['NORAD44835', date_str + '20:07:30', date_str + '20:11:00', -4.65],
    #                 ['NORAD17129', date_str + '20:12:42', date_str + '20:17:00', -1.23],
    #                 ['NORAD44797', date_str + '20:18:21', date_str + '20:24:00', 43.48],
    #                 ['NORAD10095', date_str + '20:25:24', date_str + '20:27:24', 66.59],
    #                 ['NORAD44813', date_str + '20:31:27', date_str + '20:37:27', 2.03],
    #                 ['NORAD9662',  date_str + '20:40:59', date_str + '20:42:50', 65.42],
    #                 ['NORAD41621', date_str + '20:43:13', date_str + '20:45:00', 44.05],
    #                 ['NORAD15595', date_str + '20:46:01', date_str + '20:48:01', 28.05],
    #                 ['NORAD17912', date_str + '20:49:08', date_str + '20:51:08', 54.85],
    #                 ['NORAD41635', date_str + '20:53:18', date_str + '21:00:18', 5.23]]

    OBSERVATIONS = [['NORAD40420', date_str + '15:59:02', date_str + '16:01:02', 51.50],
                    ['NORAD17974', date_str + '16:03:01', date_str + '16:05:01', 33.5],
                    ['NORAD447', date_str + '16:07:40', date_str + '16:19:40', 42.12],
                    ['NORAD21263', date_str + '16:21:34', date_str + '16:25:34', 31.53],
                    ['NORAD40358', date_str + '16:28:51', date_str + '16:32:51', 57.15],
                    ['NORAD11788', date_str + '16:34:10', date_str + '16:36:10', 51.13],
                    ['NORAD36605', date_str + '16:38:41', date_str + '16:44:41', 66.28],
                    ['NORAD37673', date_str + '16:46:51', date_str + '16:48:51', 33.75],
                    ['NORAD15099', date_str + '16:50:47', date_str + '16:59:47', 28.64]]

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

import datetime
import logging as log
import time
from pybirales.repository.message_broker import RedisManager
import dateutil.parser
import humanize
import pytz
from pybirales.services.scheduler.observation import ScheduledCalibrationObservation, ScheduledObservation


def monitor_worker(scheduler):
    """
    Start the monitoring thread

    :param scheduler: The sched instance
    :return:
    """
    time_counter = 0
    while not scheduler.empty():
        # Process every N iterations
        if time_counter % 60 == 0:
            now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
            for event in scheduler.queue:
                observation = event.argument[0]
                h_time_remaining = humanize.naturaldelta(now - observation.start_time_padded)
                h_duration = humanize.naturaldelta(observation.duration)
                log.info('The %s for the `%s` observation is scheduled to start in %s and will run for %s',
                         observation.pipeline_name,
                         observation.name,
                         h_time_remaining, h_duration)

        time_counter += 1
        time.sleep(1)

    log.info('Monitoring thread terminated')


def obs_listener_worker(obs_queue):
    """
     Listen for new scheduled observations

    :param obs_queue: The sched instance
    :return:
    """

    obs_chl = 'birales_scheduled_obs'

    # The redis instance
    _redis = RedisManager.Instance().redis

    # The PubSub interface of the redis instance
    _pubsub = _redis.pubsub()

    # Subscribe to the observations channel
    _pubsub.subscribe(obs_chl)

    for message in _pubsub.listen():
        if message['data'] == 'KILL':
            log.info('Scheduled observations listener un-subscribed from {}'.format(obs_chl))
            break
        else:
            if message['type'] == 'message':
                obs = message['data']
                so = ScheduledObservation(name=obs['name'],
                                          obs_type=obs['type'],
                                          config_file=obs['config_file'],
                                          pipeline_name=obs['pipeline'],
                                          params=obs['config_parameters'])
                obs_queue.add_observation(so)

    log.info('Observation listener terminated')

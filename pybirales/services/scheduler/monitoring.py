import datetime
import logging as log
import time
from pybirales.repository.message_broker import RedisManager
import dateutil.parser
import humanize
import pytz
from pybirales.services.scheduler.observation import ScheduledCalibrationObservation, ScheduledObservation
import json


def monitor_worker(scheduler, stop_event):
    """
    Start the monitoring thread

    :param scheduler: The sched instance
    :return:
    """
    time_counter = 0
    while not stop_event.is_set():
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


def obs_listener_worker(scheduler):
    """
     Listen for new scheduled observations

    :param scheduler: The sched instance
    :return:
    """

    obs_chl = 'birales_scheduled_obs'

    # The redis instance
    _redis = RedisManager.Instance().redis

    # The PubSub interface of the redis instance
    _pubsub = _redis.pubsub()

    # Subscribe to the observations channel
    _pubsub.subscribe(obs_chl)

    log.info('Scheduler listening on `{}` for new observations'.format(obs_chl))
    for message in _pubsub.listen():
        if message['data'] == 'KILL':
            log.info('Scheduled observations listener un-subscribed from {}'.format(obs_chl))
            break
        else:
            if message['type'] == 'message':
                log.info("New observation received by scheduler: {}".format(message['data']))

                # Create object from json string
                observation = scheduler.create_obs(json.loads(message['data']))

                # Add the scheduled objects to the queue
                scheduler.add_observation(observation)

    log.info('Observation listener terminated')

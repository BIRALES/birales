import datetime
import json
import logging as log
import time

import humanize
import pytz
from mongoengine import errors
from pybirales.events.events import InvalidObservationEvent, ObservationScheduledEvent, ObservationDeletedEvent
from pybirales.events.publisher import publish
from pybirales.repository.message_broker import pub_sub
from pybirales.repository.models import Observation
from pybirales.services.scheduler.exceptions import InvalidObservationException

OBS_CREATE_CHL = 'birales_scheduled_obs'
OBS_DEL_CHL = 'birales_delete_obs'


def monitor_worker(scheduler, stop_event):
    """
    Start the monitoring thread

    :param scheduler: The sched instance
    :return:
    """
    time_counter = 0

    log.debug('Monitoring thread started')

    while not stop_event.is_set():
        # Process every N iterations
        if time_counter % 60 == 0:
            now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
            for event in scheduler.queue:
                observation = event.argument[0]
                delta = now - observation.start_time_padded
                h_time_remaining = humanize.naturaldelta(delta)
                h_duration = humanize.naturaldelta(observation.duration)
                log.info('The %s for the `%s` observation is scheduled to start in %s and will run for %s',
                         observation.pipeline_name,
                         observation.name,
                         h_time_remaining, h_duration)

            if len(scheduler.queue) < 1:
                log.info('No queued observations.')

        time_counter += 1
        time.sleep(1)

    log.info('Monitoring thread terminated')


def obs_listener_worker(scheduler):
    """
     Listen for new scheduled observations

    :param scheduler: The sched instance
    :return:
    """

    # Subscribe to the observations channels (create and delete)
    pub_sub.subscribe([OBS_CREATE_CHL, OBS_DEL_CHL])

    log.info('Scheduler listening on `{}` for new observations'.format(OBS_CREATE_CHL))
    for message in pub_sub.listen():
        if message['data'] == 'KILL':
            log.info('KILL Received. Scheduled observations listener un-subscribed from {}'.format(OBS_CREATE_CHL))
            break

        if message['type'] != 'message':
            continue

        data = json.loads(message['data'])

        if message['channel'] == OBS_CREATE_CHL:
            log.info("New observation received by scheduler: {}".format(data))

            # Create object from json string
            observation = scheduler.create_obs(data)

            # Add the scheduled objects to the queue
            try:
                scheduler.schedule.add_observation(observation)
            except InvalidObservationException:
                log.warning('Observation is not valid')

                # Report back to front-end
                publish(InvalidObservationEvent(observation))
            else:
                publish(ObservationScheduledEvent(observation))
        if message['channel'] == OBS_DEL_CHL:
            log.debug('Delete observation %s message received.', data)

            try:
                obs_id = data['obs_id']

                observation = Observation.objects.get(id=obs_id)
                observation.delete()
            except Exception:
                log.exception('Observation could not be deleted (%s)', data)
            else:
                publish(ObservationDeletedEvent(observation))

                # Reload the schedule
                scheduler.reload()
    log.info('Observation listener terminated')

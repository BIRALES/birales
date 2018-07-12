import json
import logging as log
import sys
from pybirales.events.events import InvalidObservationEvent, ObservationScheduledEvent, ObservationDeletedEvent,SystemErrorEvent
from pybirales.events.publisher import publish
from pybirales.repository.message_broker import pub_sub
from pybirales.repository.models import Observation
from pybirales.services.scheduler.exceptions import InvalidObservationException

OBS_CREATE_CHL = 'birales_scheduled_obs'
OBS_DEL_CHL = 'birales_delete_obs'


def obs_listener_worker(scheduler):
    """
     Listen for new scheduled observations

    :param scheduler:
    :return:
    """

    # Subscribe to the observations channels (create and delete)
    channels = [OBS_CREATE_CHL, OBS_DEL_CHL]
    pub_sub.subscribe(channels)

    log.info('Scheduler listening on `{}` for new observations'.format(OBS_CREATE_CHL))
    for message in pub_sub.listen():
        if message['data'] == 'KILL' and (message['channel'] in channels) :
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
            except InvalidObservationException as e:
                # Report back to front-end
                publish(InvalidObservationEvent(observation, e.msg))
            except Exception:
                log.warning('Observation could not be added to the scheduler')
                publish(SystemErrorEvent(reason=sys.exc_info()[0]))
            else:
                publish(ObservationScheduledEvent(observation))
        if message['channel'] == OBS_DEL_CHL:
            log.debug('Delete observation %s message received.', data)

            try:
                observation = Observation.objects.get(id=data['obs_id'])
                observation.delete()
            except Exception:
                log.warning('Observation could not be deleted (%s)', data)
                publish(SystemErrorEvent(reason=sys.exc_info()[0]))
            else:
                publish(ObservationDeletedEvent(observation))

                # Reload the schedule
                scheduler.reload()
    log.info('Observation listener terminated')

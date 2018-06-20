import logging as log
import json
from pybirales.repository.message_broker import RedisManager
from pybirales.utilities.singleton import Singleton
from pybirales.repository.models import Event


def publish(event):
    """
    Publish utility function at module level
    :param event: The event that you would like to publish
    :return:
    """

    _publisher.publish(event)


@Singleton
class EventsPublisher:
    def __init__(self):
        self._redis = RedisManager.Instance().redis

    @staticmethod
    def _format_msg(payload):
        return json.dumps(payload)

    def publish(self, event):
        """
        Publish hte event to the message broker
        :param event:
        :return:
        """
        for channel in event.channels:
            s = self._redis.publish(channel, self._format_msg(event.payload))
            log.debug('{} published to channels: {} across {} subscribers'.format(event.name, event.channels, s))

        # Save the event to the database
        self.save(event)

    def save(self, event):
        """
        Save the event to the database

        :param event:
        :return:
        """

        event_model = Event(
            name=event.name,
            channels=event.channels,
            description=event.description,
            body=event.payload['body'],
            header=event.payload['header']
        )

        # Persist
        event_model.save()

_publisher = EventsPublisher.Instance()
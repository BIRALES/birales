import logging as log
import json
from pybirales.repository.message_broker import RedisManager
from pybirales.utilities.singleton import Singleton


@Singleton
class EventsPublisher:
    def __init__(self):
        self._redis = RedisManager.Instance().redis

    @staticmethod
    def _format_msg(payload):
        return json.dumps(payload)

    def publish(self, event):
        for channel in event.channels:
            s = self._redis.publish(channel, self._format_msg(event.payload))
            log.debug('{} published to channels: {} across {} subscribers'.format(event.name, event.channels, s))

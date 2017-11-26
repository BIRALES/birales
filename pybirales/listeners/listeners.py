import threading
import logging as log


class Listener(threading.Thread):
    def __init__(self, redis, channels):
        threading.Thread.__init__(self)
        self._redis = redis
        self._channels = channels
        self._pubsub = self._redis.pubsub()
        self._pubsub.subscribe(channels)

    def run(self):
        for item in self._pubsub.listen():
            if item['data'] == "KILL":
                self._pubsub.unsubscribe()
                log.info('{} listener, un-subscribed from channels: {}'.format(self.name, self._channels))
                break
            else:
                self.handle(item)

    @property
    def name(self):
        return self.__class__.__name__

    def handle(self, data):
        pass


class NotificationsListener(Listener):
    def __init__(self, redis, channels):
        Listener.__init__(self, redis, channels)

    def handle(self, data):
        # Send notifications to Slack
        pass


class SpaceDebrisListener(Listener):
    def __init__(self, redis, channels):
        Listener.__init__(self, redis, channels)

    def handle(self, data):
        # Send space debris candidates over socket-io
        pass


class LogMessagesListener(Listener):
    def __init__(self, redis, channels):
        Listener.__init__(self, redis, channels)

    def handle(self, data):
        # Send space debris candidates over socket-io
        pass

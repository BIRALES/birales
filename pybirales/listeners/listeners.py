import abc
import logging as log
import os
import threading

from slackclient import SlackClient

from pybirales.events.events import ObservationScheduledEvent
from pybirales.repository.message_broker import RedisManager
import json
import pickle


class Listener(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        # The channels the listener will be subscribed to
        self._channels = []

        # The redis instance
        self._redis = RedisManager.Instance().redis

        # The PubSub interface of the redis instance
        self._pubsub = self._redis.pubsub()

        # Stop event. When set, the thread will quit
        self._stop_event = threading.Event()

        self._name = 'Listener'

    def stop(self):
        self._stop_event.set()

    def run(self):
        log.info('Starting listener.')
        if not self._channels:
            log.warning('Listener is not subscribed to any event.'.format(self.name))
            return

        self._pubsub.subscribe(self._channels[0])

        for item in self._pubsub.listen():
            if item['data'] == "KILL" or self._stop_event.is_set():
                self._pubsub.unsubscribe()
                log.info('Listener, un-subscribed from channels: {}'.format(self.name, self._channels))
                break
            else:
                self.handle(item)

        log.info('Listener, stopped listening on these channels: {}'.format(self.name, self._channels))

    @property
    def name(self):
        if self._name:
            return self._name
        return self.__class__.__name__

    @staticmethod
    def _get_message(json_string):
        return json.loads(json_string)

    @abc.abstractmethod
    def handle(self, data):
        pass


class NotificationsListener(Listener):
    def __init__(self):
        Listener.__init__(self)

        self._channels += ObservationScheduledEvent.channels
        self.api_end_point = 'chat.postMessage'
        self.channel = '#birales'

        try:
            self.token = os.environ["SLACK_BOT_TOKEN"]
        except KeyError:
            log.warning('Slack token not set. Please set the SLACK_BOT_TOKEN env variable. Notifications disabled.')
            self.stop()
        else:
            self.slack_client = SlackClient(self.token)

        self._name = 'Notifications'

    def handle(self, data):
        if data['type'] == 'message':
            # Send notifications to Slack
            log.debug('{} listener received a message on #notifications'.format(self.name))
            msg = self._get_message(data['data'])

            request_body = dict(channel=self.channel, text=self._format_msg(msg))
            response = self.slack_client.api_call(
                self.api_end_point,
                **request_body
            )

            if response['ok']:
                log.debug('Slack message was sent successfully')
            else:
                log.warning('Slack message failed with the following error: {}'.format(response['error']))

    @staticmethod
    def _format_msg(msg):
        return '*{}*: {}'.format(msg['header']['origin'], msg['body'])


class SpaceDebrisListener(Listener):
    def __init__(self):
        Listener.__init__(self)

    def handle(self, data):
        log.debug('{} listener received {}'.format(self.name, data))
        # Send space debris candidates over socket-io
        pass

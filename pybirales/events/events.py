import logging as log
import json
import socket

STATUS_TYPE = [
    'info',
    'success',
    'error'
]


class Event:
    channels = []
    description = 'Event Description'

    def __init__(self):
        self._level = 'info'
        self.payload = {'header': {'level': self._level,
                                   'channels': self.channels,
                                   'description': self.description,
                                   'origin': socket.gethostname()
                                   },
                        'body': {}}

    def to_json(self):
        if self.payload:
            return json.dumps(self.payload)

    @property
    def name(self):
        return self.__class__.__name__


class ObservationScheduledEvent(Event):
    """
    Event representing a scheduled observation
    """

    channels = ['notifications']
    description = 'An observation was scheduled'

    def __init__(self, observation):
        """

        :param observation: An object of type ScheduledObservation
        :type observation: ScheduledObservation
        """

        Event.__init__(self)

        event = 'The `{}` observation was added to the schedule.'.format(observation.name)

        self.payload['body'] = event + '\n' + observation.start_message()

        log.info(event)

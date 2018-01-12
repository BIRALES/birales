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

        event_msg = '`{}` was _added_ to the schedule.'.format(observation.name)

        self.payload['body'] = event_msg + observation.start_message()

        log.debug(event_msg)


class ObservationScheduledCancelledEvent(Event):
    """
    Event representing a scheduled observation
    """

    channels = ['notifications']
    description = 'A scheduled observation was cancelled'

    def __init__(self, observation):
        """

        :param observation: An object of type ScheduledObservation
        :type observation: ScheduledObservation
        """

        Event.__init__(self)

        self.payload['body'] = '`{}` was _cancelled_ from the schedule.'.format(observation.name)

        log.debug(self.payload['body'])


class ObservationStartedEvent(Event):
    """
    Event is fired when an observation is started
    """

    channels = ['notifications']
    description = 'An observation was started'

    def __init__(self, observation, pipeline_name):
        """

        :param observation: An object of type ScheduledObservation
        :type observation: ScheduledObservation
        """

        Event.__init__(self)

        self.payload['body'] = '`{}` (using the `{}` pipeline) was started on *{}*'.format(observation.name,
                                                                                             pipeline_name,
                                                                                             socket.gethostname())
        log.debug(self.payload['body'])

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

        self.payload['body'] = event_msg + ' ' + observation.start_message()

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
        :param pipeline_name: The name of the pipeline
        :param observation: An object of type ScheduledObservation
        :type observation: ScheduledObservation
        """

        Event.__init__(self)

        self.payload['body'] = '`{}` (using the `{}` pipeline) was started on *{}*'.format(observation.name,
                                                                                           pipeline_name,
                                                                                           socket.gethostname())
        log.debug(self.payload['body'])


class ObservationFinishedEvent(Event):
    """
    Event is fired when an observation has finished (successfully)
    """

    channels = ['notifications']
    description = 'An observation has finished'

    def __init__(self, observation, pipeline_name):
        """
        :param pipeline_name: The name of the pipeline
        :param observation: The observation which has finished
        """

        Event.__init__(self)

        log.info('Observation {} (using the {}) finished'.format(observation.name, pipeline_name))
        self.payload['body'] = '`{}` finished successfully'.format(observation.name)
        log.debug(self.payload['body'])


class SpaceDebrisClusterDetectedEvent(Event):
    """
    Event is fired when a valid space debris cluster is detected
    """

    channels = ['notifications']
    description = 'A new beam candidate was found'

    def __init__(self, n_clusters, beam_id):
        """

        :param n_clusters: Number of valid space debris clusters
        :param beam_id: The beam id in which the space debris cluster was detected
        """

        Event.__init__(self)

        self.payload['body'] = 'A new detection cluster (`{}` data points) was made in beam {}'.format(n_clusters,
                                                                                                       beam_id)
        log.debug(self.payload['body'])


class SpaceDebrisDetectedEvent(Event):
    """
    Event is fired when a valid space debris cluster is detected
    """

    channels = ['notifications']
    description = 'A new space debris candidate was found'

    def __init__(self, sd_track):
        """

        :param sd_track: Space Debris Track detected across N beams
        :type  sd_track: SpaceDebrisTrack
        """

        Event.__init__(self)

        self.payload['body'] = 'A new space debris detection ({}) (score: `{}`, n:`{}`) was made.'.format(id(sd_track),
                                                                                                          sd_track.score,
                                                                                                          sd_track.size)
        log.debug(self.payload['body'])

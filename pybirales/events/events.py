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

    def __init__(self, obs_name, pipeline_name):
        """
        :param pipeline_name: The name of the pipeline
        :param observation: An object of type ScheduledObservation
        :type observation: ScheduledObservation
        """

        Event.__init__(self)

        self.payload['body'] = '`{}` (using the `{}` pipeline) was started on *{}*'.format(obs_name,
                                                                                           pipeline_name,
                                                                                           socket.gethostname())
        log.debug(self.payload['body'])


class ObservationFinishedEvent(Event):
    """
    Event is fired when an observation has finished (successfully)
    """

    channels = ['notifications']
    description = 'An observation has finished'

    def __init__(self, obs_name, pipeline_name):
        """
        :param pipeline_name: The name of the pipeline
        :param observation: The observation which has finished
        """

        Event.__init__(self)

        log.info('Observation {} (using the {}) finished'.format(obs_name, pipeline_name))
        self.payload['body'] = '`{}` finished successfully'.format(obs_name)
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


class TrackCreatedEvent(Event):
    """
    Event is fired when a valid space debris cluster is detected
    """

    channels = ['notifications']
    description = 'A new track candidate was found'

    def __init__(self, sd_track):
        """

        :param sd_track: Space Debris Track detected across N beams
        :type  sd_track: SpaceDebrisTrack
        """

        Event.__init__(self)

        self.payload['body'] = 'Track {} was `created` (score: {:0.3f}, size:{}).'.format(
            id(sd_track),
            sd_track.r_value,
            sd_track.size)
        log.debug(self.payload['body'])


class TrackModifiedEvent(Event):
    """
    Event is fired when a valid space debris cluster is detected
    """

    channels = ['notifications']
    description = 'An existing track was updated'

    def __init__(self, sd_track):
        """

        :param sd_track: Space Debris Track detected across N beams
        :type  sd_track: SpaceDebrisTrack
        """

        Event.__init__(self)

        self.payload['body'] = 'Track {} was `modified` (score: {:0.3f}, size:{}).'.format(
            id(sd_track),
            sd_track.r_value,
            sd_track.size)
        log.debug(self.payload['body'])


class CalibrationRoutineStartedEvent(Event):
    """
    Event is fired when the calibration routine starts
    """

    channels = ['notifications']
    description = 'Generating calibration coefficients'

    def __init__(self, obs_name, corr_matrix_filepath):
        """
        :param corr_matrix_filepath: The filepath of the correlation matrix
        :param obs_name: The name of the observation
        :type corr_matrix_filepath: String
        :type obs_name: String
        """

        Event.__init__(self)

        msg = 'Using the correlation matrix ({}) to generate the calibration coefficients from the `{}` observation' \
            .format(corr_matrix_filepath, obs_name)
        self.payload['body'] = msg

        log.debug(self.payload['body'])


class CalibrationRoutineFinishedEvent(Event):
    """
    Event is fired when the calibration routine finishes
    """

    channels = ['notifications']
    description = 'Calibration coefficients generated'

    def __init__(self, obs_name, coeffs_dir):
        """
        :param coeffs_dir: The filepath were the calibration coefficients will be generated
        :param obs_name: The name of the observation
        :type coeffs_dir: String
        :type obs_name: String
        """

        Event.__init__(self)

        msg = 'The `{}` observation\'s calibration coefficients were generated at {}'.format(obs_name, coeffs_dir)
        self.payload['body'] = msg

        log.debug(self.payload['body'])


class TrackCandidatesFoundEvent(Event):
    """
    Event is fired when the post processing finishes
    """

    channels = ['notifications']
    description = 'Post-processing of the observation finished and N candidates were found'

    def __init__(self, n_candidates, obs_name):
        """

        :param n_candidates: The number of candidates found
        :param obs_name: The name of the observation
        """

        Event.__init__(self)

        msg = '{} tracks were found in observation `{}`'.format(n_candidates, obs_name)
        self.payload['body'] = msg

        log.debug(self.payload['body'])


class InvalidObservationEvent(Event):
    """
    Event representing an error upon trying to schedule an observation
    """

    channels = ['notifications']
    description = 'The observation is not valid.'

    def __init__(self, observation):
        """

        :param observation: An object of type ScheduledObservation
        :type observation: ScheduledObservation
        """

        Event.__init__(self)

        self.payload['body'] = 'Something went wrong. The observation ({}) could not be added to the scheduler.'.format(
            observation.name)

        log.debug(self.payload['body'])


class ObservationDeletedEvent(Event):
    """
    Event representing a successful deletion of an observation
    """

    channels = ['notifications']
    description = 'The observation was deleted.'

    def __init__(self, observation):
        """

        :param observation: The observation that was deleted (mongoengine model)
        :type observation: Observation
        """

        Event.__init__(self)

        self.payload['body'] = 'Observation {} was deleted successfully.'.format(observation.name)

        log.debug(self.payload['body'])


class BIRALESSchedulerReloadedEvent(Event):
    """
    Event is triggered when the BIRALES scheduler service is reloaded.
    """

    channels = ['notifications']
    description = 'BIRALES scheduler service was reloaded.'

    def __init__(self):
        Event.__init__(self)

        self.payload['body'] = 'The BIRALES system was reloaded'

        log.debug(self.payload['body'])

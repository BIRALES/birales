import datetime
import logging as log

import dateutil.parser
import humanize
import pytz

from pybirales.base.observation_manager import ObservationManager, CalibrationObservationManager
from pybirales.pipeline.base.definitions import PipelineBuilderIsNotAvailableException
from pybirales.pipeline.pipeline import AVAILABLE_PIPELINES_BUILDERS
from pybirales.repository.models import Observation as ObservationModel
from pybirales.services.instrument.best2 import pointing_time
from pybirales.services.scheduler.exceptions import ObservationScheduledInPastException, IncorrectObservationParameters
from pybirales.utilities.source_transit import get_calibration_source_declination, get_calibration_sources


class ScheduledObservation(object):
    """
    Represents any observation created and scheduled by the user
    """
    TYPE = 'observation'
    DEFAULT_WAIT_SECONDS = 5
    OBS_END_PADDING = datetime.timedelta(seconds=60)  # The default time padding at the end of the observation
    OBS_START_PADDING = datetime.timedelta(seconds=60)  # The default time padding at the start of the observation
    RECALIBRATION_TIME = datetime.timedelta(hours=24)  # Recalibrate every 24 hours

    def __init__(self, name, pipeline_name, config_parameters, config_file, model_id=None):
        """
        Initialisation function for the Scheduled Observation

        :param name: The name of the observation
        :param obs_type:
        :param pipeline_name:
        :param config_parameters:
        :param config_file:
        """

        self.name = name
        self.pipeline_name = pipeline_name
        self.config_file = config_file
        self.parameters = config_parameters

        if self.pipeline_name not in AVAILABLE_PIPELINES_BUILDERS:
            raise PipelineBuilderIsNotAvailableException(pipeline_name, AVAILABLE_PIPELINES_BUILDERS)

        self.declination = self._declination()
        self.duration = self._duration()
        self.start_time = self._start_time()
        self.created_at = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        self.end_time = None
        self.pipeline_builder = None

        if self.duration:
            self.end_time = self.start_time + self.duration

        # Sched event instance associated with this observation
        self.event = None

        self._next_observation = None
        self._prev_observation = None

        if not model_id:
            self.model = ObservationModel(
                name=self.name, date_time_start=self.start_time,
                date_time_end=self.end_time,
                pipeline=self.pipeline_name,
                type=self.TYPE,
                status='pending',
                config_parameters=self.parameters,
                config_file=self.config_file
            )
        else:
            self.model = ObservationModel.objects.get(id=model_id)

        self.manager = ObservationManager()

    @property
    def id(self):
        return self.model.id

    def save(self):
        self.model.save()

    def _declination(self):
        """
        Set the declination for this observation
        :return:
        """

        try:
            declination = self.parameters['beamformer']['reference_declination']
        except KeyError:
            log.warning('Reference declination is not specified for observation `{}`'.format(self.name))

            return None
        else:
            return declination

    def _duration(self):
        """
        Set the duration of this observation
        :return:
        """

        try:
            duration = datetime.timedelta(seconds=self.parameters['duration'])
        except KeyError:
            raise IncorrectObservationParameters('Duration was not specified for observation `{}`'.format(self.name))
        else:
            return duration

    def _start_time(self):
        """
        Set the start time of this observation
        :return:
        """
        try:
            start_time = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
            if 'start_time' in self.parameters:
                start_time = self.parameters['start_time']

            if not isinstance(start_time, datetime.datetime):
                start_time = dateutil.parser.parse(start_time)

                if not start_time.tzinfo:
                    raise IncorrectObservationParameters(
                        'Invalid timezone for start date ({}) of observation'.format(start_time))
        except ValueError:
            raise IncorrectObservationParameters('Invalid start time for observation `{}`'.format(self.name))
        except KeyError:
            raise IncorrectObservationParameters('Start time not specified for observation `{}`'.format(self.name))
        else:
            return start_time

    @property
    def wait_time(self):
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        return (self.start_time_padded - now).total_seconds()

    @property
    def is_calibration_obs(self):
        return isinstance(self, ScheduledCalibrationObservation)

    @property
    def next_observation(self):
        """
        Get the next observation
        :return:
        """
        return self._next_observation

    @next_observation.setter
    def next_observation(self, observation):
        """

        :param observation:
        :return:
        """
        if self.declination and observation:
            # The padding of the observation following this one needs to be updated
            time_delta = datetime.timedelta(seconds=pointing_time(dec1=self.declination, dec2=observation.declination))
            time_delta_min = time_delta.total_seconds() / 60

            observation.start_time_padding = time_delta

            log.debug(
                'Adding a padding of {:0.2f} minutes for observation `{}` to account for the movement of the antenna '
                'from {:0.2f} DEC to {:0.2f} DEC'.format(time_delta_min,
                                                         self.name,
                                                         self.declination,
                                                         observation.declination))

        # Set the next observation
        self._next_observation = observation

    @property
    def prev_observation(self):
        """

        :return:
        """

        return self._prev_observation

    @prev_observation.setter
    def prev_observation(self, observation):
        """

        :param observation:
        :return:
        """
        self._prev_observation = observation

    def start_message(self):
        """
        Compose a human friendly start message for the observation start time

        :return:
        """
        start_msg = "The `{}` observation, using the `{}`, is scheduled".format(self.name, self.pipeline_name)

        if self.start_time:
            start_msg += " to run at {:%Y-%m-%d %H:%M:%S} UTC".format(self.start_time)
        else:
            start_msg += " to start NOW."

        if self.end_time:
            start_msg += ' and will run for {:0.2f} minutes.'.format(self.duration.total_seconds() / 60.)
        else:
            start_msg += ' and will run indefinitely.'

        return start_msg

    @property
    def start_time_padded(self):
        """
        Padded start time accounts for the time needed to point the array and initialise the pipeline

        :return:
        """
        return self.start_time + ScheduledObservation.OBS_START_PADDING

    @start_time_padded.setter
    def start_time_padded(self, padding):
        self._start_time_padding = padding

    def update_start_time_padding(self, prev_observation):
        time_delta = self._start_time_padding
        time_delta += datetime.timedelta(seconds=pointing_time(prev_observation.declination, self.declination))

        self.start_time_padded = time_delta

        log.debug(
            'Adding a padding of {} for observation `{}` to account for the movement of the antenna '
            'from {} DEC to {} DEC'.format(humanize.naturaldelta(time_delta), self.name, prev_observation.declination,
                                           self.declination))

    @property
    def end_time_padded(self):
        """
        Padded end time accounts for the time needed to shutdown the pipeline

        :return:
        """
        return self.end_time + ScheduledObservation.OBS_END_PADDING

    def is_in_future(self):
        """
        Check that this observation is in the future

        :return:
        """

        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        wait_seconds = (self.start_time_padded - now).total_seconds()

        if wait_seconds < 1:
            raise ObservationScheduledInPastException(self)

    def is_calibration_needed(self, obs):
        """
        Traverse the schedule and return the first valid calibration observation
        that meets the minimum recalibration time criteria

        :param obs: An observation node in the schedule
        :return:
        """
        if obs is None:
            return True

        if isinstance(obs, ScheduledCalibrationObservation):
            # Time between observation and calibration routine
            time_to_calibration = (self.start_time_padded - obs.end_time)

            # Check that it is less than the minimum re-calibration time
            if time_to_calibration < self.RECALIBRATION_TIME:
                return False

        # Get next observation is schedule
        return self.is_calibration_needed(obs.next_observation)


class ScheduledCalibrationObservation(ScheduledObservation):
    """
    Represents the calibration observations that are scheduled by the user.
    """

    TYPE = 'calibration'

    def __init__(self, name, pipeline_name, config_parameters, config_file, model_id=None):
        pipeline_name = 'correlation_pipeline'

        ScheduledObservation.__init__(self, name, pipeline_name, config_parameters, config_file, model_id)

        self.manager = CalibrationObservationManager()

    def is_calibration_needed(self, obs):
        """
        Overridden function from Parent. Given that self is a calibration instance, fetch the next
        observation in schedule

        :param obs:
        :return:
        """
        if obs is None:
            return True

        return self.is_calibration_needed(obs.next_observation)


class ScheduledAutoCalibrationObservation(ScheduledCalibrationObservation):
    """
    Represents calibration observations that are scheduled automatically by the BIRALES
    service

    """
    CALIBRATION_TIME = 3600
    CALIBRATION_SOURCES = get_calibration_sources()

    def __init__(self, source, start_time, config_file):
        name = '{}_{:%Y-%m-%dT%H:%M}.calib'.format(source, start_time)
        params = {
            'start_time': start_time,
            'duration': self.CALIBRATION_TIME,
        }

        if name in self.CALIBRATION_SOURCES:
            params['beamformer']['reference_declination'] = get_calibration_source_declination(source)
        else:
            raise IncorrectObservationParameters('Calibration source `{}` is not valid'.format(source))

        ScheduledCalibrationObservation.__init__(self, name, 'correlation_pipeline', params, config_file)

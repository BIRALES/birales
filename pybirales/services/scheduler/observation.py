import datetime
import logging as log

import dateutil.parser
import pytz

from pybirales.pipeline.base.definitions import PipelineBuilderIsNotAvailableException
from pybirales.pipeline.pipeline import AVAILABLE_PIPELINES_BUILDERS
from pybirales.repository.models import Observation as ObservationModel
from pybirales.services.scheduler.exceptions import ObservationScheduledInPastException, IncorrectObservationParameters


class ScheduledObservation(object):
    """
    Represents any observation created and scheduled by the user
    """
    TYPE = 'observation'
    DEFAULT_WAIT_SECONDS = 5
    OBS_END_PADDING = datetime.timedelta(seconds=60)  # The default time padding at the end of the observation
    OBS_START_PADDING = datetime.timedelta(seconds=60)  # The default time padding at the start of the observation
    RECALIBRATION_TIME = datetime.timedelta(hours=24)  # Recalibrate every 24 hours

    def __init__(self, name=None, pipeline_name=None, config_parameters=None, config_file=None, model_id=None):
        """
        Initialisation function for the Scheduled Observation

        :param name: The name of the observation
        :param obs_type:
        :param pipeline_name:
        :param config_parameters:
        :param config_file:
        """

        if not model_id:
            # if model id is not given, the parameters are required
            if not all([name, pipeline_name, config_parameters, config_file]):
                raise IncorrectObservationParameters('Some observation parameters are missing for new observation')

            self.name = name
            self.pipeline_name = pipeline_name
            self.config_file = config_file
            self.parameters = config_parameters
            self.duration = self._duration()
            self.start_time = self._start_time()

            self.end_time = self.start_time + self.duration
            self.created_at = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

            self.model = ObservationModel(
                name=self.name,
                date_time_start=self.start_time,
                date_time_end=self.end_time,
                pipeline=self.pipeline_name,
                type=self.TYPE,
                status='pending',
                config_parameters=self.parameters,
                config_file=self.config_file
            )
        else:
            self.model = ObservationModel.objects.get(id=model_id)
            self.pipeline_name = self.model.pipeline
            self.name = self.model.name
            self.start_time = self.model.date_time_start.replace(tzinfo=pytz.utc)
            self.end_time = self.model.date_time_end.replace(tzinfo=pytz.utc)
            self.TYPE = self.model.type
            self.status = self.model.status
            self.parameters = self.model.config_parameters
            self.config_file = self.model.config_file
            self.duration = self._duration()
            self.created_at = self.model.created_at

        if self.pipeline_name not in AVAILABLE_PIPELINES_BUILDERS:
            raise PipelineBuilderIsNotAvailableException(self.pipeline_name, AVAILABLE_PIPELINES_BUILDERS)

        self.declination = self._declination()

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
        return (self.start_time - now).total_seconds()

    @property
    def is_calibration_obs(self):
        return isinstance(self, ScheduledCalibrationObservation)

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

    def is_in_future(self):
        """
        Check that this observation is in the future

        :return:
        """

        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        wait_seconds = (self.start_time - now).total_seconds()

        if wait_seconds < 1:
            raise ObservationScheduledInPastException(self)

    @property
    def should_start(self):
        """

        :return:
        """
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        return (now - self.start_time).total_seconds() > 0


class ScheduledCalibrationObservation(ScheduledObservation):
    """
    Represents the calibration observations that are scheduled by the user.
    """

    TYPE = 'calibration'

    def __init__(self, name=None, pipeline_name=None, config_parameters=None, config_file=None, model_id=None):
        pipeline_name = 'correlation_pipeline'

        ScheduledObservation.__init__(self, name, pipeline_name, config_parameters, config_file, model_id)

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

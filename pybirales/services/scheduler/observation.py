import datetime

import pytz
from pybirales.services.instrument.best2 import BEST2
from pybirales.services.scheduler.exceptions import ObservationScheduledInPastException
from pybirales.pipeline.pipeline import CorrelatorPipelineManagerBuilder


class ScheduledObservation:
    DEFAULT_WAIT_SECONDS = 5
    OBS_END_PADDING = datetime.timedelta(seconds=60)
    OBS_START_PADDING = datetime.timedelta(seconds=60)

    # Recalibrate every 24 hours
    RECALIBRATION_TIME = datetime.timedelta(hours=24)

    def __init__(self, name, config_file, pipeline_builder, dec, start_time, duration=None):
        """
        Initialisation function for the Scheduled Observation

        :param name: The name of the observation
        """

        self.name = name
        self.config_file = config_file
        self.pipeline_name = pipeline_builder.manager.name
        self.declination = dec
        self.start_time = start_time
        self.duration = duration
        self.end_time = None

        self.pipeline_builder = pipeline_builder
        self.created_at = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

        self._next_observation = None
        self._prev_observation = None

        # Set the default time padding at the end of the observation
        self.end_time_padding = ScheduledObservation.OBS_END_PADDING

        # Set the default time padding at the start of the observation
        self.start_time_padding = ScheduledObservation.OBS_START_PADDING

        if self.duration:
            self.end_time = start_time + self.duration

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

        # The padding of the observation following this one needs to be updated
        observation.start_time_padding = BEST2().pointing_time(dec1=self.declination, dec2=observation.declination)

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
        start_msg = "Observation {}, using the {} is scheduled to start NOW".format(self.name, self.pipeline_name)

        if self.start_time:
            start_msg += " to run at {:%Y-%m-%d %H:%M:%S}".format(self.start_time)
        else:
            start_msg += " to start NOW"

        if self.end_time:
            start_msg += ' and will run for {} seconds'.format(self.duration)
        else:
            start_msg += ' and will run indefinitely'

        return start_msg

    @property
    def start_time_padded(self):
        """
        Padded start time accounts for the time needed to point the array and initialise the pipeline

        :return:
        """
        return self.start_time + self.start_time_padding

    @property
    def end_time_padded(self):
        """
        Padded end time accounts for the time needed to shutdown the pipeline

        :return:
        """
        return self.end_time + self.end_time_padded

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
    def __init__(self, source, config_file):
        name = '{}_{}.calibration'.format(source['name'], source['date'])
        dec = source['parameters']['dec']
        start_time = source['transit_time']
        duration = 3600

        pipeline_builder = CorrelatorPipelineManagerBuilder()

        ScheduledObservation.__init__(self, name, config_file, pipeline_builder, dec, start_time,
                                      duration)

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



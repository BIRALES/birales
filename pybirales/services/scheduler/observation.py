import datetime

import pytz
from pybirales.services.instrument.best2 import BEST2
from pybirales.services.scheduler.exceptions import ObservationScheduledInPastException
from pybirales.pipeline.pipeline import CorrelatorPipelineManagerBuilder


class ScheduledObservation:
    DEFAULT_WAIT_SECONDS = 5
    OBS_END_PADDING = datetime.timedelta(seconds=60)
    OBS_START_PADDING = datetime.timedelta(seconds=60)

    def __init__(self, name, config_file, pipeline_builder, dec, start_time, duration=None, nxt_obs=None, prv_obs=None):
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

        self._next_observation = nxt_obs
        self._prev_observation = prv_obs

        # Set the default time padding at the end of the observation
        self.end_time_padding = ScheduledObservation.OBS_END_PADDING

        # Set the default time padding at the start of the observation
        self.start_time_padding = ScheduledObservation.OBS_START_PADDING

        if prv_obs:
            # Set the time taken to move the instrument from the previous declination to this observation declination
            # todo this need to be changed every time a previous observation is added
            self.start_time_padding = BEST2().pointing_time(dec1=prv_obs.declination, dec2=self.declination)

        if self.duration:
            self.end_time = start_time + self.duration

    @property
    def next_observation(self):
        return self._next_observation

    @next_observation.setter
    def next_observation(self, observation):
        self._next_observation = observation

    @property
    def prev_observation(self):
        return self._prev_observation

    @prev_observation.setter
    def prev_observation(self, observation):
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
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        wait_seconds = (self.start_time_padded - now).total_seconds()

        if wait_seconds < 1:
            raise ObservationScheduledInPastException(self)


class ScheduledCalibrationObservation(ScheduledObservation):
    def __init__(self, source, config_file, prv_obs):
        name = '{}_{}.calibration'.format(source['name'], source['date'])
        dec = source['parameters']['dec']
        start_time = source['transit_time']
        duration = 3600

        ScheduledObservation.__init__(self, name, config_file, CorrelatorPipelineManagerBuilder(), dec, start_time,
                                      duration, prv_obs)

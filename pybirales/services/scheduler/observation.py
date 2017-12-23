from pybirales.services.scheduler.exceptions import PipelineIsNotAvailableException
from pybirales.pipeline.pipeline import DetectionPipelineMangerBuilder, CorrelatorPipelineManagerBuilder, \
    StandAlonePipelineMangerBuilder
import datetime
import dateutil.parser
import pytz


class ScheduledObservation:
    DEFAULT_WAIT_SECONDS = 5
    OBS_END_PADDING = datetime.timedelta(seconds=60)
    OBS_START_PADDING = datetime.timedelta(seconds=60)

    def __init__(self, name, config_file, pipeline_name, dec, start_time, duration=None, start_time_padding=None):
        """
        Initialisation function for the Scheduled Observation

        :param name: The name of the observation
        """

        self.name = name
        self.config_file = config_file
        self.pipeline_name = pipeline_name
        self.declination = dec
        self.start_time = start_time
        self.duration = duration
        self.end_time = None

        self.pipeline_builder = self._get_builder(self.pipeline_name)
        self.created_at = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        self.wait_seconds = ScheduledObservation.DEFAULT_WAIT_SECONDS

        self.end_time_padding = ScheduledObservation.OBS_END_PADDING
        self.start_time_padding = ScheduledObservation.OBS_START_PADDING

        if start_time_padding:
            self.start_time_padding = start_time_padding

        if self.duration:
            self.end_time = start_time + self.duration

        """
        if 'duration' in self.parameters:
            self.end_time = dateutil.parser.parse(self.parameters['start_time']) + datetime.timedelta(
                seconds=self.parameters['duration'])
            self.duration = self.parameters['duration']

            self.end_time_padded = self.end_time + ScheduledObservation.OBS_END_PADDING

        if 'start_time' in self.parameters:
            self.start_time = dateutil.parser.parse(self.parameters['start_time'])

            # todo - This should be set dynamically depending on previous pointing
            self.start_time_padded = self.start_time - ScheduledObservation.OBS_START_PADDING
        """

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

    @staticmethod
    def _get_builder(pipeline_name):
        """
        Return the pipeline manager builder by name

        @todo - this can converted into a util class an be accessible by any module
        :param pipeline_name: The name of the pipeline builder
        :raises PipelineIsNotAvailableException: The pipeline with the supplied pipeline_name does not exist
        :return:
        """

        if pipeline_name == 'detection_pipeline':
            return DetectionPipelineMangerBuilder()
        elif pipeline_name == 'correlation_pipeline':
            return CorrelatorPipelineManagerBuilder()
        elif pipeline_name == 'standalone_pipeline':
            return StandAlonePipelineMangerBuilder()

        raise PipelineIsNotAvailableException(pipeline_name)

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


class ScheduledCalibrationObservation(ScheduledObservation):
    def __init__(self, name, config_file, dec, start_time, start_time_padding=None):

        pipeline_name = 'correlator_pipeline'
        duration = 3600
        super(ScheduledCalibrationObservation, self).__init__(name, config_file, pipeline_name, dec, start_time,
                                                              duration, start_time_padding)

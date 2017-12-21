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

    def __init__(self, name, params):
        """
        Initialisation function for the Scheduled Observation

        :param name: The name of the observation
        :param params: The configuration parameters associated with this observation
        """

        self.name = name
        self.pipeline_name = params['pipeline_name']
        self.config_file = params['config_file']
        self.parameters = params['config_parameters']
        self.pipeline_builder = self._get_builder(self.pipeline_name)
        self.created_at = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        self.wait_seconds = ScheduledObservation.DEFAULT_WAIT_SECONDS

        self.start_time = None
        self.end_time = None
        self.duration = None

        # Padded start time accounts for the time needed to point the array
        self.start_time_padded = None

        # Padded end time accounts for the time needed to shutdown the pipeline
        self.end_time_padded = None

        if 'duration' in self.parameters:
            self.end_time = dateutil.parser.parse(self.parameters['start_time']) + datetime.timedelta(
                seconds=self.parameters['duration'])
            self.duration = self.parameters['duration']

            self.end_time_padded = self.end_time + ScheduledObservation.OBS_END_PADDING

        if 'start_time' in self.parameters:
            self.start_time = dateutil.parser.parse(self.parameters['start_time'])

            # todo - This should be set dynamically depending on previous pointing
            self.start_time_padded = self.start_time - ScheduledObservation.OBS_START_PADDING

    def start_message(self):
        """
        Compose a human friendly start message for the observation start time

        :return:
        """
        start_msg = "Observation {}, using the {} is scheduled to start NOW".format(
            self.name, self.parameters['pipeline'])

        if self.start_time:
            start_msg += " to run at {:%Y-%m-%d %H:%M:%S}".format(self.start_time)
        else:
            start_msg += " to start NOW"

        if self.end_time:
            start_msg += ' and will run for {} seconds'.format(self.parameters['duration'])
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


class ScheduledCalibrationObservation(ScheduledObservation):
    def __init__(self):
        super(ScheduledCalibrationObservation, self).__init__()
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.parameters = None

import datetime
import logging as log

from pybirales.base.controller import BackendController, InstrumentController
from pybirales.birales_config import BiralesConfig
from pybirales.events.events import ObservationStartedEvent, ObservationFinishedEvent, CalibrationRoutineStartedEvent, \
    CalibrationRoutineFinishedEvent
from pybirales.events.publisher import publish
from pybirales.pipeline.pipeline import get_builder_by_id, CorrelatorPipelineManagerBuilder
from pybirales.services.calibration.calibration import CalibrationFacade
from pybirales.services.post_processing.processor import PostProcessor
from pybirales import settings


class ObservationManager:

    def __init__(self):
        """

        """
        self._post_processor = PostProcessor()
        self._instrument_control = None
        self._backend_control = None

    def _pre_process(self, observation):
        self._instrument_control = InstrumentController()

        self._backend_control = BackendController()

        self.obs_config = BiralesConfig(config_file_path=observation.config_file,
                                        config_options=observation.parameters)

        self.obs_config.load()

    def run(self, observation):
        """

        :param observation:
        :return:
        """

        self._pre_process(observation)

        observation.model.settings = self.obs_config.to_dict()
        observation.save()

        publish(ObservationStartedEvent(observation.name, observation.pipeline_name))

        pipeline_builder = get_builder_by_id(observation.pipeline_name)

        pipeline_builder.build()

        pipeline_builder.manager.start_pipeline(duration=observation.duration.total_seconds())

        observation.model.date_time_end = datetime.datetime.utcnow()
        observation.save()

        publish(ObservationFinishedEvent(observation.name, observation.pipeline_name))

        self._post_process(observation)

        self.tear_down()

    def _post_process(self, observation):
        """

        :param observation:
        :return:
        """
        log.info('Post-processing observation. Generating output files.')
        self._post_processor.process(observation._model)
        log.info('Post-processing of the observation finished')

    def tear_down(self):
        log.debug('Stopping the instrument')
        self._instrument_control.stop()

        log.debug('Stopping the Backend')
        self._backend_control.stop()


class CalibrationObservationManager(ObservationManager):

    def __init__(self):
        """

        """
        self._calibration_facade = CalibrationFacade()

        ObservationManager.__init__(self)

    def __pre_process(self, observation):
        # Call the parent's pre_process function
        self._pre_process(observation)

        self.calibration_dir, self.corr_matrix_filepath = self._calibration_facade.get_calibration_filepath()

        new_options = {'observation': {'type': 'calibration'}, 'corrmatrixpersister':
            {'corr_matrix_filepath': self.corr_matrix_filepath}}

        log.debug('Correlation matrix will be saved at %s', self.corr_matrix_filepath)

        self.obs_config.update_config(new_options)

    def run(self, observation):
        """

        :param observation:
        :return:
        """

        self.__pre_process(observation)

        observation.model.settings = self.obs_config.to_dict()
        observation.save()

        publish(ObservationStartedEvent(observation.name, observation.pipeline_name))

        self._instrument_control.point(observation.declination)

        pipeline_builder = CorrelatorPipelineManagerBuilder()

        pipeline_builder.build()

        pipeline_builder.manager.start_pipeline(duration=observation.duration.total_seconds())

        observation.model.date_time_end = datetime.datetime.utcnow()
        observation.save()

        publish(ObservationFinishedEvent(observation.name, observation.pipeline_name))

        # Use the correlation matrix to calibrate the system
        self.calibrate(observation, self.calibration_dir, self.corr_matrix_filepath)

        # Terminate (gracefully) the connection to the BEST instrument and the ROACH backend
        self.tear_down()

    def calibrate(self, observation, calibration_dir, corr_matrix_filepath):
        """

        :param observation:
        :param calibration_dir:
        :param corr_matrix_filepath:
        :return:
        """
        publish(CalibrationRoutineStartedEvent(observation.name, corr_matrix_filepath))

        # Run the calibration routine
        self._calibration_facade.calibrate(calibration_dir, corr_matrix_filepath)

        publish(CalibrationRoutineFinishedEvent(observation.name, calibration_dir))

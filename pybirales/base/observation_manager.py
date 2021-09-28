import logging as log
import os

from pybirales import settings
from pybirales.base.controller import BackendController, InstrumentController
from pybirales.birales_config import BiralesConfig
from pybirales.events.events import ObservationStartedEvent, ObservationFinishedEvent, CalibrationRoutineStartedEvent, \
    CalibrationRoutineFinishedEvent, ObservationFailedEvent, CalibrationObservationFailedEvent
from pybirales.events.publisher import publish
from pybirales.pipeline.base.definitions import CalibrationFailedException
from pybirales.pipeline.base.definitions import PipelineError, BEST2PointingException
from pybirales.pipeline.modules.persisters.corr_matrix_persister import create_corr_matrix_filepath
from pybirales.pipeline.pipeline import get_builder_by_id, CorrelatorPipelineManagerBuilder
from pybirales.services.calibration.calibration import CalibrationFacade
from pybirales.services.post_processing.processor import PostProcessor
from pybirales.services.scheduler.exceptions import SchedulerException


class ObservationManager:

    def __init__(self):
        """

        """
        self._post_processor = None
        self._instrument_control = None
        self._backend_control = None

    def _pre_process(self, observation):
        """
        Run any pre-processing routines for this observation

        :param observation: The observation to pre-process
        :return:
        """

        self.obs_config = BiralesConfig(config_file_path=observation.config_file, config_options=observation.parameters)

        self.obs_config.load()

        self._post_processor = PostProcessor()

        self._instrument_control = InstrumentController()

        self._backend_control = BackendController()

        # Make sure observation is in the database
        observation.save()

        self.obs_config.update_config({'observation': {'id': observation.id}})

    def run(self, observation):
        """
        Run the observation (the pipeline and any post-processing routines)

        :param observation: The observation to run
        :return:
        """

        try:
            self._pre_process(observation)

            observation.model.settings = self.obs_config.to_dict()
        except PipelineError as e:
            log.exception("A fatal error has occurred whilst trying to run %s (%s)", observation.name, str(e))
            publish(ObservationFailedEvent(observation, str(e)))

            observation.model.status = 'failed'
            observation.save()

            self.tear_down()

            return

        observation.model.status = 'running'

        try:
            # Point the instrument to the desired declination
            self._instrument_control.point(observation.declination)

            # Read the current declination of the antenna
            observation.model.antenna_dec = self._instrument_control.get_declination()
        except BEST2PointingException:
            publish(ObservationFailedEvent(observation, "Failed to point antenna."))

        observation.save()

        publish(ObservationStartedEvent(observation.name, observation.pipeline_name))

        pipeline_builder = get_builder_by_id(observation.pipeline_name)

        try:
            pipeline_builder.build()
            pipeline_builder.manager.start_pipeline(duration=observation.duration.total_seconds(),
                                                    observation=observation)
        except SchedulerException:
            log.exception("A fatal error has occurred whilst trying to run %s", observation.name)
            publish(ObservationFailedEvent(observation, "A scheduler exception has occurred"))

            observation.model.status = 'failed'
            observation.save()
        except PipelineError:
            log.exception("An fatal error has occurred whilst trying to run %s", observation.name)
            publish(ObservationFailedEvent(observation, "A pipeline error has occurred"))

            observation.model.status = 'failed'
            observation.save()
        else:
            publish(ObservationFinishedEvent(observation.name, observation.pipeline_name))

            # if settings.detection.save_candidates or settings.detection.save_tdm:
            #     self._post_process(observation)
        # self.obs_config.db_disconnect()
        self.tear_down()

    def _post_process(self, observation):
        """
        Run the post-processing routines for this observation

        :param observation: The observation to post process
        :return:
        """
        log.info('Post-processing observation. Generating output files.')
        self._post_processor.process(observation.model)
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

    def __pre_process(self, observation, correlation_matrix_filepath=None):
        """
        Calibration-specific pre-processing routines

        :param observation:
        :return:
        """

        # Call the parent's pre_process function
        self._pre_process(observation)

        # self._calibration_facade = CalibrationFacade(correlation_matrix_filepath)

        if correlation_matrix_filepath:
            self.calibration_dir = os.path.dirname(correlation_matrix_filepath)
            self.corr_matrix_filepath = correlation_matrix_filepath
        else:

            if settings.manager.offline:
                # if offline calibration
                self.calibration_dir = os.path.dirname(settings.rawdatareader.filepath)
                self.corr_matrix_filepath = os.path.join(self.calibration_dir, observation.name + '__corr.h5')
            else:
                # if online calibration

                self.corr_matrix_filepath =  create_corr_matrix_filepath()
                self.calibration_dir = os.path.dirname(self.corr_matrix_filepath)

        new_options = {'observation': {'type': 'calibration'}, 'corrmatrixpersister':
            {'corr_matrix_filepath': self.corr_matrix_filepath}}

        log.debug('Correlation matrix will be saved at %s', self.corr_matrix_filepath)

        self.obs_config.update_config(new_options)

    def _run_corr_pipeline(self, observation):
        """

        :param observation:
        :return:
        """

        observation.model.settings = self.obs_config.to_dict()
        observation.model.status = 'running'
        observation.save()

        publish(ObservationStartedEvent(observation.name, observation.pipeline_name))

        try:
            # Point the instrument to the desired declination
            self._instrument_control.point(observation.declination)

            # Read the current declination of the antenna
            observation.model.antenna_dec = self._instrument_control.get_declination()
        except BEST2PointingException:
            publish(ObservationFailedEvent(observation, "Failed to point antenna."))

        pipeline_builder = CorrelatorPipelineManagerBuilder()

        try:
            pipeline_builder.build()
            pipeline_builder.manager.start_pipeline(duration=observation.duration.total_seconds(),
                                                    observation=observation)
        except SchedulerException:
            log.exception("An fatal error has occurred whilst trying to run %s", observation.name)
            publish(ObservationFailedEvent(observation, "A scheduler exception has occurred. Calibration Failed."))

            observation.model.status = 'failed'
            observation.save()
        except PipelineError:
            log.exception("An fatal error has occurred whilst trying to run %s", observation.name)
            publish(ObservationFailedEvent(observation, "A pipeline error has occurred. Calibration Failed."))

            observation.model.status = 'failed'
            observation.save()
        else:
            publish(ObservationFinishedEvent(observation.name, observation.pipeline_name))

            return self.corr_matrix_filepath
        return None

    def run(self, observation, corr_matrix_filepath=None):
        """
        The calibration observation to run

        :param observation:
        :return:
        """
        self.__pre_process(observation, corr_matrix_filepath)

        if not corr_matrix_filepath:
            corr_matrix_filepath = self._run_corr_pipeline(observation)

        if corr_matrix_filepath:
            try:
                # Use the correlation matrix to calibrate the system
                self._calibrate(observation, corr_matrix_filepath)
            except CalibrationFailedException:
                observation.model.status = 'failed'
                observation.save()
                publish(CalibrationObservationFailedEvent(observation, "Calibration observation failed"))

                raise
            else:
                observation.model.status = 'finished'
                observation.save()

                publish(CalibrationRoutineFinishedEvent(observation, self.calibration_dir))
        else:
            publish(CalibrationObservationFailedEvent(observation, "Correlation pipeline failed to "
                                                                   "generate a valid correlation matrix"))
        # Terminate (gracefully) the connection to the BEST instrument and the ROACH backend
        self.tear_down()

    def _calibrate(self, observation, corr_matrix_filepath):
        """
        Run the calibration routine using the calibration facade

        :param observation:
        :param corr_matrix_filepath:
        :return:
        """
        observation.model.status = 'calibrating'
        observation.save()

        publish(CalibrationRoutineStartedEvent(observation.name, corr_matrix_filepath))

        # Run the calibration routine
        try:
            real, imag, fringe_image = self._calibration_facade.calibrate(self.calibration_dir, corr_matrix_filepath)
        except IOError as e:
            log.exception("An IO error has occured")
            raise CalibrationFailedException(e)
        # except IndexError as e:
        #     raise CalibrationFailedException(e)
        else:
            observation.model.real = real.tolist()
            observation.model.imag = imag.tolist()
            observation.model.fringe_image = fringe_image

            observation.save()

import logging as log

from pybirales.base import settings
from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule
from pybirales.blobs.channelised_data import ChannelisedBlob
from pybirales.modules.detection.detection_strategies import SpaceDebrisDetection
from pybirales.modules.detection.queue import BeamCandidatesQueue
from pybirales.modules.detection.repository import BeamCandidateRepository
from pybirales.modules.detection.repository import ConfigurationRepository
from pybirales.modules.detection.detection_strategy import DetectionStrategy


class Detector(ProcessingModule):
    def __init__(self, config, input_blob=None):
        if type(input_blob) is not ChannelisedBlob:
            raise PipelineError(
                "Detector: Invalid input data type, should be BeamformedBlob, DummyBlob or ReceiverBlob")

        # Load detection algorithm dynamically (specified in config file)
        self.detection_strategy = SpaceDebrisDetection(DetectionStrategy())

        # Repository Layer for saving the beam candidates to the Data store
        self._candidates_repository = BeamCandidateRepository()

        # Repository Layer for saving the configuration to the Data store
        self._configurations_repository = ConfigurationRepository()

        # Data structure that hold the detected debris (for merging)
        self._debris_queue = BeamCandidatesQueue(settings.beamformer.nbeams)

        # Flag that indicates whether the configuration was persisted
        self._config_persisted = False

        super(Detector, self).__init__(config, input_blob)

        self.name = "Detector"

    def process(self, obs_info, input_data, output_data):
        """
        Run the Space Debris Detector pipeline
        :return void
        """

        # Checks if input data is empty
        # if not input_data.any():
        #     log.warning('Input data is empty')
        #     return

        if not self._config_persisted:
            self._configurations_repository.persist(obs_info)
            self._config_persisted = True

        # Process the beam data to detect the beam candidates
        new_beam_candidates = self.detection_strategy.detect(obs_info, input_data)

        for new_beam_candidate in new_beam_candidates:
            self._debris_queue.enqueue(new_beam_candidate)

        log.info('%s beam candidates, were found', len(new_beam_candidates))

    def generate_output_blob(self):
        pass

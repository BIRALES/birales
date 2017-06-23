import logging as log
import time
import numpy as np

from pybirales.base import settings
from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule
from pybirales.blobs.channelised_data import ChannelisedBlob
from pybirales.modules.detection.queue import BeamCandidatesQueue
from pybirales.modules.detection.repository import BeamCandidateRepository
from pybirales.modules.detection.repository import ConfigurationRepository
from pybirales.modules.detection.strategies.m_dbscan import MultiProcessingDBScanDetectionStrategy, init
from pybirales.modules.detection.strategies.strategies import SpaceDebrisDetection
from multiprocessing import Pool


class Detector(ProcessingModule):
    def __init__(self, config, input_blob=None):
        if type(input_blob) is not ChannelisedBlob:
            raise PipelineError(
                "Detector: Invalid input data type, should be BeamformedBlob, DummyBlob or ReceiverBlob")

        # Load detection algorithm dynamically (specified in config file)
        self.detection_strategy = SpaceDebrisDetection(MultiProcessingDBScanDetectionStrategy())

        # Repository Layer for saving the beam candidates to the Data store
        self._candidates_repository = BeamCandidateRepository()

        # Repository Layer for saving the configuration to the Data store
        self._configurations_repository = ConfigurationRepository()

        # Data structure that hold the detected debris (for merging)
        self._debris_queue = BeamCandidatesQueue(settings.beamformer.nbeams)

        # Flag that indicates whether the configuration was persisted
        self._config_persisted = False

        self.pool = Pool(8, init, ())

        self.counter = 0

        self.noise = []

        self.mean_noise = 0

        super(Detector, self).__init__(config, input_blob)

        self.name = "Detector"

    def _get_noise_estimation(self, input_data):
        if self.counter < settings.detection.n_noise_samples:
            power = np.power(np.abs(input_data[0, :, settings.detection.noise_channels, :]), 2)

            if settings.detection.noise_use_rms:
                #  use RMS
                noise = np.sqrt(np.mean(np.power(power, 2)))
            else:
                # use mean
                noise = np.mean(power)

            self.noise.append(noise)
            self.mean_noise = np.mean(self.noise)
        return float(self.mean_noise)

    def process(self, obs_info, input_data, output_data):
        """
        Run the Space Debris Detector pipeline
        :return void
        """

        # estimate the noise from the data
        obs_info['noise'] = self._get_noise_estimation(input_data)

        if not self._config_persisted:
            self._configurations_repository.persist(obs_info)
            self._config_persisted = True

        # Process the beam data to detect the beam candidates
        tt = time.time()
        new_beam_candidates = self.detection_strategy.detect(obs_info, input_data, self.pool, self._debris_queue)
        log.debug('Candidates detected in %0.3f s', time.time() - tt)

        log.info('%s beam candidates, were found', len(new_beam_candidates))

        self.counter += 1

    def generate_output_blob(self):
        pass

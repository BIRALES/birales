import logging as log
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool
from pybirales.modules.detection.detection_strategies import SpaceDebrisDetection
from pybirales.modules.detection.repository import BeamCandidateRepository
from pybirales.modules.detection.repository import DataSetRepository
from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule
from pybirales.base import settings
from pybirales.blobs.beamformed_data import BeamformedBlob
from pybirales.blobs.channelised_data import ChannelisedBlob
from pybirales.blobs.dummy_data import DummyBlob
from pybirales.blobs.receiver_data import ReceiverBlob
from pybirales.modules.detection.beam import Beam


class Detector(ProcessingModule):
    def __init__(self, config, input_blob=None):
        log.info('Initialising the Space Debris Detector Module')

        if type(input_blob) not in [BeamformedBlob, DummyBlob, ReceiverBlob, ChannelisedBlob]:
            raise PipelineError("PFB: Invalid input data type, should be BeamformedBlob, DummyBlob or ReceiverBlob")

        # Load detection algorithm dynamically (specified in config file)
        self.detection_strategy = SpaceDebrisDetection(settings.detection.detection_strategy)

        super(Detector, self).__init__(config, input_blob)
        self.name = "Detector"

    def generate_output_blob(self):
        pass

    @staticmethod
    def _create_beam(obs_info, n_beam, input_blob):
        log.debug('Generating beam %s from the input data', n_beam)
        beam = Beam(beam_id=n_beam,
                    dec=0.0,
                    ra=0.0,
                    ha=0.0,
                    top_frequency=0.0,
                    frequency_offset=0.0,
                    obs_info=obs_info, beam_data=input_blob)
        return beam

    def process(self, obs_info, input_data, output_data):
        """
        Run the Space Debris Detector pipeline
        :return void
        """
        # Checks if input data is empty
        if not input_data.any():
            log.warning('Input data is empty')
            return

        # Extract beams from the data set we are processing
        log.info('Extracting beam data')

        beams = []
        for n_beam in range(0, settings.beamformer.nbeams):
            beams.append(self._create_beam(obs_info, n_beam, input_data))
        log.debug(input_data.shape)
        # Process the beam data to detect the beam candidates
        if settings.detection.nthreads > 1:
            log.info('Running space debris detection algorithm on %s beams in parallel', len(beams))
            beam_candidates = self._get_beam_candidates_parallel(beams)
        else:
            log.info('Running space debris detection algorithm on %s beams in serial mode', len(beams))
            beam_candidates = self._get_beam_candidates_single(beams)

        log.info('Data processed, saving %s beam candidates to database', len(beam_candidates))
        log.debug("Input data: %s shape: %s", np.sum(input_data),  input_data.shape)


        # self._save_data_set()

        # self._save_beam_candidates(beam_candidates)

    def _get_beam_candidates_single(self, beams):
        """
        Run the detection algorithm using 1 process
        :return: beam_candidates Beam candidates detected across the 32 beams
        """
        beam_candidates = []
        for beam in beams:
            beam_candidates += self._detect_space_debris_candidates(beam)

        return beam_candidates

    def _get_beam_candidates_parallel(self, beams):
        """

        :return: beam_candidates Beam candidates detected across the 32 beams
        """

        # Initialise thread pool with
        pool = ThreadPool(settings.detection.nthreads)

        # Run N threads
        beam_candidates = pool.map(self._detect_space_debris_candidates, beams)

        # Flatten list of beam candidates returned by the N threads
        beam_candidates = [candidate for sub_list in beam_candidates for candidate in sub_list]

        # Close thread pool upon completion
        pool.close()
        pool.join()

        return beam_candidates

    def _save_beam_candidates(self, beam_candidates):
        if settings.monitoring.save_candidates:
            # Persist beam candidates to database
            beam_candidates_repository = BeamCandidateRepository(self.data_set)
            beam_candidates_repository.persist(beam_candidates)

    def _save_data_set(self):
        if settings.detection.save_data_set:
            # Persist data_set to database
            data_set_repository = DataSetRepository()
            data_set_repository.persist(self.data_set)

    def _detect_space_debris_candidates(self, beam):
        # Save raw beam data for post processing
        if beam.id in settings.monitoring.visualize_beams:
            beam.visualize('raw beam ' + str(beam.id))

        # Apply the pre-processing filters to the beam data
        beam.apply_filters()

        # Save the filtered beam data to the database
        if settings.monitoring.save_filtered_beam_data:
            beam.save_detections()

        # Save filtered beam data for post processing
        if beam.id in settings.monitoring.visualize_beams:
            beam.visualize('filtered beam ' + str(beam.id))

        # Run detection algorithm on the beam data to extract possible candidates
        candidates = self.detection_strategy.detect(beam)

        log.info('%s candidates were detected in beam %s', len(candidates), beam.id)

        return candidates

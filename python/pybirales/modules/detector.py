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
from pybirales.plotters.spectrogram_plotter import plotter
from pybirales.modules.detection.queue import BeamCandidatesQueue

import time


class Detector(ProcessingModule):
    def __init__(self, config, input_blob=None):
        if type(input_blob) not in [BeamformedBlob, DummyBlob, ReceiverBlob, ChannelisedBlob]:
            raise PipelineError(
                "Detector: Invalid input data type, should be BeamformedBlob, DummyBlob or ReceiverBlob")

        # Load detection algorithm dynamically (specified in config file)
        self.detection_strategy = SpaceDebrisDetection(settings.detection.detection_strategy)

        # Repository Layer for saving the beam candidates to the Data store
        self._candidates_repository = BeamCandidateRepository()

        # Data structure that hold the detected debris (for merging)
        self._debris_queue = BeamCandidatesQueue(settings.beamformer.nbeams)

        # Initialise thread pool with N threads
        self._thread_pool = ThreadPool(settings.detection.nthreads)

        super(Detector, self).__init__(config, input_blob)

        self.name = "Detector"

    def generate_output_blob(self):
        pass

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
        log.debug('Extracting beam data')

        # Create the beams
        beams = [Beam(beam_id=n_beam, obs_info=obs_info, beam_data=input_data)
                 for n_beam in range(settings.beamformer.nbeams)]

        # Process the beam data to detect the beam candidates
        if settings.detection.nthreads > 1:
            new_beam_candidates = self._get_beam_candidates_parallel(beams)
        else:
            new_beam_candidates = self._get_beam_candidates_single(beams)

        for new_beam_candidate in new_beam_candidates:
            if new_beam_candidate:
                self._debris_queue.enqueue(new_beam_candidate)

        if settings.detection.save_candidates:
            # Persist beam candidates to database
            self._debris_queue.save()

    def _get_beam_candidates_single(self, beams):
        """
        Run the detection algorithm using 1 process

        :return: beam_candidates Beam candidates detected across the 32 beams
        """

        log.debug('Running space debris detection algorithm on %s beams in serial', len(beams))


        # Get the detected beam candidates
        s = time.time()
        beam_candidates = [self._detect_space_debris_candidates(beam) for beam in beams]
        log.info('detections in %0.4f seconds', time.time() - s)

        # Do not add beam candidates that a
        return [candidate for sub_list in beam_candidates for candidate in sub_list if candidate]

    def _get_beam_candidates_parallel(self, beams):
        """
        Run the detection algorithm using N threads

        :return: beam_candidates Beam candidates detected across the 32 beams
        """

        log.debug('Running space debris detection algorithm on %s beams in parallel', len(beams))

        # Run using N threads
        beam_candidates = self._thread_pool.map(self._detect_space_debris_candidates, beams)

        # Close thread pool upon completion
        # self._thread_pool.close()
        # self._thread_pool.join()

        # Flatten list of beam candidates returned by the N threads
        return [candidate for sub_list in beam_candidates for candidate in sub_list if candidate]

    def _detect_space_debris_candidates(self, beam):
        # plotter.plot(beam.snr, 'detection/input_beam_6_' + str(time.time()), beam.id == 6)

        # Apply the pre-processing filters to the beam data
        beam.apply_filters()

        # Run detection algorithm on the beam data to extract possible candidates
        log.debug('Running detection algorithm on beam %s', beam.id)
        candidates = self.detection_strategy.detect(beam)

        log.debug('%s candidates were detected in beam %s', len(candidates), beam.id)

        return candidates

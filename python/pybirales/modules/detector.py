import logging as log
import os

from multiprocessing.dummy import Pool as ThreadPool
from pybirales.modules.detection.detection_strategies import SpaceDebrisDetection, SpiritSpaceDebrisDetectionStrategy
from pybirales.modules.detection.repository import BeamCandidateRepository
from pybirales.modules.detection.repository import ConfigurationRepository
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
from multiprocessing import Pool


def dd2(beam):
    # Apply the pre-processing filters to the beam data
    candidates = []
    try:
        beam.apply_filters()
        # Run detection algorithm on the beam data to extract possible candidates
        candidates = SpiritSpaceDebrisDetectionStrategy().detect(beam)
    except Exception:
        log.exception('Something went wrong with process')
    return candidates


def dd3(q, beam):
    # Apply the pre-processing filters to the beam data
    candidates = []
    try:
        beam.apply_filters()

        # Run detection algorithm on the beam data to extract possible candidates
        candidates = SpiritSpaceDebrisDetectionStrategy().detect(beam)
    except Exception:
        log.exception('Something went wrong with process')

    new_beam_candidates = [candidate for sub_list in candidates for candidate in sub_list if candidate]

    for new_beam_candidate in new_beam_candidates:
        if new_beam_candidate:
            q.enqueue(new_beam_candidate)

    if settings.detection.save_candidates:
        # Persist beam candidates
        q.save()

    log.info('%s beam candidates, were found', len(new_beam_candidates))

    return candidates


class Detector(ProcessingModule):
    def __init__(self, config, input_blob=None):
        if type(input_blob) not in [BeamformedBlob, DummyBlob, ReceiverBlob, ChannelisedBlob]:
            raise PipelineError(
                "Detector: Invalid input data type, should be BeamformedBlob, DummyBlob or ReceiverBlob")

        # Load detection algorithm dynamically (specified in config file)
        self.detection_strategy = SpaceDebrisDetection(settings.detection.detection_strategy)

        # Repository Layer for saving the beam candidates to the Data store
        self._candidates_repository = BeamCandidateRepository()

        # Repository Layer for saving the configuration to the Data store
        self._configurations_repository = ConfigurationRepository()

        # Data structure that hold the detected debris (for merging)
        self._debris_queue = BeamCandidatesQueue(settings.beamformer.nbeams)

        # Initialise thread pool with N threads
        self._thread_pool = ThreadPool(settings.detection.nthreads)

        self._m_pool = Pool()

        # Flag that indicates whether the configuration was persisted
        self._config_persisted = False

        super(Detector, self).__init__(config, input_blob)

        self.name = "Detector"

        self.pool = Pool(processes=4)

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

        if not self._config_persisted:
            self._configurations_repository.persist(obs_info)
            self._config_persisted = True

        # Create the beams
        beams = [Beam(beam_id=n_beam, obs_info=obs_info, beam_data=input_data)
                 for n_beam in range(settings.detection.beam_range[0], settings.detection.beam_range[1])]

        # Process the beam data to detect the beam candidates
        if settings.detection.multi_proc:
            new_beam_candidates = self._get_beam_candidates_multi_process(beams)
        elif settings.detection.nthreads > 1:
            new_beam_candidates = self._get_beam_candidates_parallel(beams)
        else:
            new_beam_candidates = self._get_beam_candidates_single(beams)

        for new_beam_candidate in new_beam_candidates:
            if new_beam_candidate:
                self._debris_queue.enqueue(new_beam_candidate)

        if settings.detection.save_candidates:
            # Persist beam candidates
            self._debris_queue.save()

        log.info('%s beam candidates, were found', len(new_beam_candidates))

    def _get_beam_candidates_single(self, beams):
        """
        Run the detection algorithm using 1 process

        :return: beam_candidates Beam candidates detected across the 32 beams
        """

        log.debug('Running space debris detection algorithm on %s beams in serial', len(beams))

        # Get the detected beam candidates
        beam_candidates = [self._detect_space_debris_candidates(beam) for beam in beams]

        # Do not add beam candidates that a
        return [candidate for sub_list in beam_candidates for candidate in sub_list if candidate]

    def _get_beam_candidates_multi_process(self, beams):
        beam_candidates = []

        # pool = self._m_pool
        try:

            beam_candidates = self.pool.map(dd2, beams)
            # results = [pool.apply_async(dd2, args=(beam,)) for beam in beams]
            # beam_candidates = [p.get() for p in results]
        except Exception:
            log.exception('An exception has occurred')

        # self.pool.close()
        # self.pool.join()

        # Flatten list of beam candidates returned by the N threads
        return [candidate for sub_list in beam_candidates for candidate in sub_list if candidate]

    def _get_beam_candidates_parallel(self, beams):
        """
        Run the detection algorithm using N threads

        :return: beam_candidates Beam candidates detected across the 32 beams
        """

        log.debug('Running space debris detection algorithm on %s beams in parallel', len(beams))

        # Run using N threads
        beam_candidates = self._thread_pool.map(self._detect_space_debris_candidates, beams)
        # Flatten list of beam candidates returned by the N threads
        return [candidate for sub_list in beam_candidates for candidate in sub_list if candidate]

    def _detect_space_debris_candidates(self, beam):
        plotter.plot(beam.snr, 'detection/input_beam_0', beam.id == 0)

        # Apply the pre-processing filters to the beam data
        beam.apply_filters()
        # Run detection algorithm on the beam data to extract possible candidates
        candidates = self.detection_strategy.detect(beam)

        return candidates


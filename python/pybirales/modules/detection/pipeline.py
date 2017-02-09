import logging as log

from multiprocessing.dummy import Pool as ThreadPool
from pybirales.modules.detection.data_set import DataSet
from pybirales.modules.detection.detection_strategies import SpaceDebrisDetection
from pybirales.modules.detection.repository import BeamCandidateRepository
from pybirales.modules.detection.repository import DataSetRepository
from pybirales.base import settings


class SpaceDebrisDetectorPipeline:
    def __init__(self, observation_name, data_set_name, n_beams):
        self.data_set = DataSet(observation_name, data_set_name, n_beams)
        log.info('Processing data set %s in %s', observation_name, data_set_name)

        # Load detection algorithm dynamically (specified in config file)
        self.detection_strategy = SpaceDebrisDetection(settings.detection.detection_strategy)
        log.info('Using %s algorithm', self.detection_strategy.name)

        self.beams_to_process = n_beams

    def run(self):
        """
        Run the Space Debris Detector pipeline
        :return void
        """

        # Extract beams from the data set we are processing
        log.info('Extracting beam data from data set %s', self.data_set.name)
        beams = self.data_set.create_beams(self.beams_to_process)

        # Process the beam data to detect the beam candidates
        if settings.detection.parallel:
            log.info('Running space debris detection algorithm on %s beams in parallel', len(beams))
            beam_candidates = self._get_beam_candidates_parallel(beams)
        else:
            log.info('Running space debris detection algorithm on %s beams in serial mode', len(beams))
            beam_candidates = self._get_beam_candidates_single(beams)

        log.info('Data processed, saving %s beam candidates to database', len(beam_candidates))

        self._save_data_set()

        self._save_beam_candidates(beam_candidates)

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
        pool = ThreadPool(16)

        # Run N threads
        beam_candidates = pool.map(self._detect_space_debris_candidates, beams)

        # Flatten list of beam candidates returned by the N threads
        beam_candidates = [candidate for sub_list in beam_candidates for candidate in sub_list]

        # Close thread pool upon completion
        pool.close()
        pool.join()

        return beam_candidates

    def _save_beam_candidates(self, beam_candidates):
        if settings.io.save_candidates:
            # Persist beam candidates to database
            beam_candidates_repository = BeamCandidateRepository(self.data_set)
            beam_candidates_repository.persist(beam_candidates)

    def _save_data_set(self):
        if settings.io.save_data_set:
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
import logging as log

from configuration.application import config
from data_set import DataSet
from detection_strategies import DBScanSpaceDebrisDetectionStrategy
from repository import BeamCandidateRepository
from repository import DataSetRepository
from multiprocessing.dummy import Pool as ThreadPool


class SpaceDebrisDetectorPipeline:
    def __init__(self, observation_name, data_set_name, n_beams):
        log.info('Processing data set %s in %s', observation_name, data_set_name)
        self.data_set = DataSet(observation_name, data_set_name, n_beams)
        self.detection_strategy = DBScanSpaceDebrisDetectionStrategy()

    def run(self):
        """
        Run the Space Debris Detector pipeline
        :return void
        """
        log.info('Running space debris detection algorithm on %s beams', len(self.data_set.beams))

        # Process the beam data to detect the beam candidates
        if config.get_boolean('application', 'PARALLEL'):
            beam_candidates = self._get_beam_candidates_parallel()
        else:
            beam_candidates = self._get_beam_candidates_single()

        log.info('Data processed, saving %s beam candidates to database', len(beam_candidates))

        self._save_data_set()

        self._save_beam_candidates(beam_candidates)

    def _get_beam_candidates_single(self):
        """
        Run the detection algorithm using 1 process
        :return: beam_candidates Beam candidates detected across the 32 beams
        """
        beam_candidates = []
        for beam in self.data_set.beams:
            beam_candidates += self._detect_space_debris_candidates(beam)

        return beam_candidates

    def _get_beam_candidates_parallel(self):
        """

        :return: beam_candidates Beam candidates detected across the 32 beams
        """

        # Initialise thread pool with
        pool = ThreadPool(4)

        # Run N threads
        beam_candidates = pool.map(self._detect_space_debris_candidates, self.data_set.beams)

        # Flatten list of beam candidates returned by the N threads
        beam_candidates = [candidate for sub_list in beam_candidates for candidate in sub_list]

        # Close thread pool upon completion
        pool.close()
        pool.join()

        return beam_candidates

    def _save_beam_candidates(self, beam_candidates):
        if config.get_boolean('io', 'SAVE_CANDIDATES'):
            # Persist beam candidates to database
            beam_candidates_repository = BeamCandidateRepository(self.data_set)
            beam_candidates_repository.persist(beam_candidates)

    def _save_data_set(self):
        if config.get_boolean('io', 'SAVE_DATA_SET'):
            # Persist data_set to database
            data_set_repository = DataSetRepository()
            data_set_repository.persist(self.data_set)

    def _detect_space_debris_candidates(self, beam):
        # Save raw beam data for post processing
        if config.get_boolean('visualization', 'VISUALIZE_BEAM'):
            beam.visualize('raw beam ' + str(beam.id))

        # Apply the pre-processing filters to the beam data
        beam.apply_filters()

        # Save the filtered beam data to the database
        if config.get_boolean('visualization', 'SAVE_FILTERED_BEAM_DATA'):
            beam.save_detections()

        # Save filtered beam data for post processing
        if config.get_boolean('visualization', 'VISUALIZE_BEAM'):
            beam.visualize('filtered beam ' + str(beam.id))

        # Run detection algorithm on the beam data to extract possible candidates
        candidates = self.detection_strategy.detect(beam)

        log.info('%s candidates were detected in beam %s', len(candidates), beam.id)
        return candidates

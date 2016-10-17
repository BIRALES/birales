import logging as log
from data_set import DataSet
from detection_strategies import DBScanSpaceDebrisDetectionStrategy
from detection_candidates import BeamSpaceDebrisCandidate
from repository import DataSetRepository
from repository import BeamCandidateRepository


class SpaceDebrisDetector:
    def __init__(self, observation_name, data_set_name):
        log.info('Processing data set %s in %s', observation_name, data_set_name)
        self.data_set = DataSet(observation_name, data_set_name)
        self.detection_strategy = DBScanSpaceDebrisDetectionStrategy()

    def run(self):
        """
        Run the Space Debris Detector pipeline
        :return void
        """
        log.info('Running space debris detection algorithm on %s beams', len(self.data_set.beams))

        beam_candidates = []
        for beam in self.data_set.beams:
            beam_candidates += self.detect_space_debris_candidates(beam)

        # Persist data_set to database
        data_set_repository = DataSetRepository()
        data_set_repository.persist(self.data_set)

        # Persist beam candidates to database
        beam_candidates_repository = BeamCandidateRepository(self.data_set)
        beam_candidates_repository.persist(beam_candidates)

    def detect_space_debris_candidates(self, beam):
        beam.apply_filters()

        log.debug('Running space debris detection algorithm on beam %s data', beam.id)
        candidates = self.detection_strategy.detect(beam)

        log.info('%s candidates were detected in beam %s', len(candidates), beam.id)
        return candidates

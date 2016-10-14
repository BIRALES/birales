import logging as log
from data_set import DataSet
from detection_strategies import DBScanSpaceDebrisDetectionStrategy
from detection_candidates import BeamSpaceDebrisCandidate
from repository import DataSetRepository


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
        n_beam_candidates = 0
        n_beams = len(self.data_set.beams)
        log.info('Running space debris detection algorithm on %s beams', n_beams)
        for beam in self.data_set.beams:
            beam_candidates = self.detect_space_debris_candidates(beam)
            n_beam_candidates += len(beam_candidates)

            # for beam_candidate in beam_candidates:
            #     beam_candidate.save()

            log.info('Beam candidates saved to database')
        log.info('%s beam space debris candidates were detected across %s beams', n_beam_candidates, n_beams)

        data_set_repository = DataSetRepository(self.data_set)
        data_set_repository.persist(self.data_set)

    def detect_space_debris_candidates(self, beam):
        beam.apply_filters()

        log.debug('Running space debris detection algorithm on beam %s data', beam.id)
        candidates = self.detection_strategy.detect(beam)

        log.info('%s candidates where detected in beam %s', len(candidates), beam.id)
        return candidates

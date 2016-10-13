import logging as log
from observation import Observation
from detection_strategies import DBScanSpaceDebrisDetectionStrategy
from detection_candidates import BeamSpaceDebrisCandidate


class SpaceDebrisDetector:
    def __init__(self, observation, data_set):
        log.info('Processing data set %s in %s', observation, data_set)
        self.observation = Observation(observation, data_set)
        self.detection_strategy = DBScanSpaceDebrisDetectionStrategy()

    def run(self):
        """
        Run the Space Debris Detector pipeline
        :return void
        """
        log.info('Running space debris detection algorithm on %s beams', len(self.observation.beams))
        n_beam_candidates = 0
        for beam in self.observation.beams:
            beam.apply_filters()
            # beam.save()

            beam_candidates = self.detect_space_debris_candidates(beam)
            n_beam_candidates += len(beam_candidates)
            # beam_candidates.save()
            # map(BeamSpaceDebrisCandidate.save, beam_candidates)
        log.info('%s beam space debris candidates were detected  algorithm on %s beams', len(self.observation.beams))

    def detect_space_debris_candidates(self, beam):
        log.debug('Running space debris detection algorithm on beam %s data', beam.id)
        candidates = self.detection_strategy.detect(beam)

        log.info('%s candidates where detected in beam %s', len(candidates), beam.id)
        return candidates

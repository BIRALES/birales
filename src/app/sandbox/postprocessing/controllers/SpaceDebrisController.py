import app.sandbox.postprocessing.config.application as config

from app.sandbox.postprocessing.lib.BeamDataFilter import Filters
from app.sandbox.postprocessing.lib.SpaceDebrisDetection import *
from app.sandbox.postprocessing.lib.DBScanSpaceDebrisDetectionStrategy import DBScanSpaceDebrisDetectionStrategy
from app.sandbox.postprocessing.models.Observation import Observation

import logging as log


class SpaceDebrisController:
    def __init__(self, observation = 'medicina_07_03_2016', data_set = 'mock_1358'):
        self.observation = Observation(observation, data_set)

    def run(self):
        log.info('Processing data set %s in %s', self.observation.data_set, self.observation.name)
        for beam in self.observation.beams:
            log.info('Analysing beam data from beam %s', beam.id)
            if config.SAVE_INPUT_DATA:
                log.info('Input beam data saved in %s', config.INPUT_BEAM_FILE_NAME)
                beam.save(file_name = config.INPUT_BEAM_FILE_NAME)

            # Pre-processing: Remove background noise from beam data
            filtered_beam = Filters.remove_background_noise(beam, 3.)
            log.info('Background noise removed from input beam %s', beam.id)

            # Pre-processing: Remove transmitter channel from beam data
            filtered_beam = Filters.remove_transmitter_channel(filtered_beam)
            log.info('Transmitter frequency removed from filtered beam %s', beam.id)

            # Post-processing: Select algorithm to use for space debris detection
            log.info('Running Space Debris Detection Algorithm')
            sdd = SpaceDebrisDetection(DBScanSpaceDebrisDetectionStrategy(max_detections = config.MAX_DETECTIONS))

            # Post-processing: Detect debris track using chosen algorithm
            candidates = sdd.detect(beam = filtered_beam)
            log.info('%s candidates where detected in beam %s', len(candidates), beam.id)

            self.save(filtered_beam, candidates)

    def save(self, filtered_beam, candidates):
        # Post-processing: Save candidates to disk
        # todo encapsulate this logic in a separate class (model?)
        if config.SAVE_INPUT_DATA:
            log.info('Filtered beam %s saved as %s', filtered_beam.id, config.FILTERED_BEAM_FILE_NAME)
            filtered_beam.save(file_name = config.FILTERED_BEAM_FILE_NAME)

        if config.VIEW_CANDIDATES:
            # Visualise detected candidates (detection profile)
            log.info('Space debris candidates were saved in %s', filtered_beam.id, self.observation.beam_output_data)
            candidates.view_candidates(output_dir = self.observation.beam_output_data, beam = filtered_beam)

        if config.SAVE_CANDIDATES:
            # Save HTML table
            log.info('Space debris candidates were saved')
            candidates.save_candidates(observation = self.observation)

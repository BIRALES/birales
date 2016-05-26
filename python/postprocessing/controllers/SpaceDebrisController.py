import config.application as config
import logging as log

from lib.BeamDataFilter import Filters
from lib.SpaceDebrisDetection import *
from lib.DBScanSpaceDebrisDetectionStrategy import DBScanSpaceDebrisDetectionStrategy
from models.Observation import Observation


# todo plot detections on original beam data
# todo fix bug why frontend is not agreeing with back end plots
# todo re implment front with client side plotting using plotly
# todo fill in remaining pickle data
# todo test using data
# todo remove old code like xml config once you are sure it working
class SpaceDebrisController:
    def __init__(self, observation='medicina_07_03_2016', data_set='mock_1358'):
        self.observation = Observation(observation, data_set)

    def run(self):
        log.info('Processing data set %s in %s', self.observation.data_set, self.observation.name)
        for beam in self.observation.beams[14:16]:
            log.info('Analysing beam data from beam %s', beam.id)
            if config.SAVE_INPUT_DATA:
                log.info('Input beam data saved in %s', config.INPUT_BEAM_FILE_NAME)
                beam.save(file_name=config.INPUT_BEAM_FILE_NAME)

            # Pre-processing: Remove background noise from beam data
            filtered_beam = Filters.remove_background_noise(beam, 3.)
            log.info('Background noise removed from input beam %s', beam.id)

            # Pre-processing: Remove transmitter channel from beam data
            filtered_beam = Filters.remove_transmitter_channel(filtered_beam)
            log.info('Transmitter frequency removed from filtered beam %s', beam.id)

            # Post-processing: Select algorithm to use for space debris detection
            log.info('Running Space Debris Detection Algorithm')
            sdd = SpaceDebrisDetection(DBScanSpaceDebrisDetectionStrategy(max_detections=config.MAX_DETECTIONS))

            # Post-processing: Detect debris track using chosen algorithm
            candidates = sdd.detect(beam=filtered_beam)

            log.info('%s candidates where detected in beam %s', len(candidates), beam.id)

            self.save(beam, filtered_beam, candidates)

    def save(self, beam, filtered_beam, candidates):
        # Post-processing: Save candidates to disk
        # todo encapsulate this logic in a separate class (model?)
        if config.SAVE_INPUT_DATA:
            log.info('Filtered beam %s saved as %s', filtered_beam.id, config.FILTERED_BEAM_FILE_NAME)
            filtered_beam.save(file_name=config.FILTERED_BEAM_FILE_NAME)

        if config.VIEW_CANDIDATES:
            # Visualise detected candidates (detection profile)
            log.info('Space debris candidates were saved as %s in %s', filtered_beam.id,
                     self.observation.beam_output_data)
            candidates.view_candidates(output_dir=self.observation.beam_output_data, beam=beam)

        if config.SAVE_CANDIDATES:
            # Save HTML table
            log.info('Space debris candidates were saved')
            candidates.save_candidates(observation=self.observation)

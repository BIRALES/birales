from app.sandbox.postprocessing.lib.BeamDataFilter import Filters
from app.sandbox.postprocessing.lib.SpaceDebrisDetection import *
from app.sandbox.postprocessing.lib.DBScanSpaceDebrisDetectionStrategy import DBScanSpaceDebrisDetectionStrategy
from app.sandbox.postprocessing.models.Beam import Beam
from app.sandbox.postprocessing.models.Observation import Observation

import app.sandbox.postprocessing.config.application as config


class SpaceDebrisController:
    def __init__(self):
        self.observation = Observation('medicina_07_03_2016', '1358')

    def run(self):
        # todo repeat for each beam
        # get input data to consume
        beam = Beam(beam_id = 15, d_delta = 1.0, dha = 1.25, observation = self.observation)

        if config.SAVE_INPUT_DATA:
            beam.save(file_name = config.INPUT_BEAM_FILE_NAME)

        # Pre-processing: Remove background noise from beam data
        filtered_beam = Filters.remove_background_noise(beam, 3.)

        # Pre-processing: Remove transmitter channel from beam data
        filtered_beam = Filters.remove_transmitter_channel(filtered_beam)

        filtered_beam.save(file_name = config.FILTERED_BEAM_FILE_NAME)

        # Post-processing: Select algorithm to use for space debris detection
        sdd = SpaceDebrisDetection(DBScanSpaceDebrisDetectionStrategy(max_detections = config.MAX_DETECTIONS))

        # Post-processing: Detect debris track using chosen algorithm
        candidates = sdd.detect(beam = filtered_beam)

        self.save(filtered_beam, candidates)

    def save(self, filtered_beam, candidates):
        # Post-processing: Save candidates to disk
        # todo encapsulate this logic in a separate class (model?)
        if config.SAVE_INPUT_DATA:
            filtered_beam.save(file_name = config.FILTERED_BEAM_FILE_NAME)

        if config.VIEW_CANDIDATES:
            # Visualise detected candidates
            candidates.view_candidates(output_dir = self.observation.beam_output_data, beam = filtered_beam)

        if config.SAVE_CANDIDATES:
            # Save HTML table
            candidates.save_candidates(output_dir = self.observation.beam_output_data)

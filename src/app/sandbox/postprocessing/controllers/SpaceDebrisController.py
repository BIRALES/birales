from app.sandbox.postprocessing.lib.BeamDataFilter import Filters
from app.sandbox.postprocessing.lib.SpaceDebrisDetection import *
from app.sandbox.postprocessing.models.Beam import Beam


import os


class SpaceDebrisController:
    def __init__(self):
        # todo parameters should be taken from a separate configuration file
        self.output_dir = os.path.join('public', 'results')
        self.observation_name = '25484'
        self.beams_output_dir = os.path.join(self.output_dir, self.observation_name, 'beams')
        self.persist_results = True

    def run(self):
        # todo repeat for each beam
        # get input data to consume
        beam = Beam(beam_id = 1, d_delta = 1.0, dha = 1.25, observation = self.observation_name)

        if self.persist_results:
            beam.save(file_name = 'input_beam')

        sdd = SpaceDebrisDetection(LineSpaceDebrisDetectionStrategy(max_detections = 3))

        # remove background noise from beam data
        filtered_beam = Filters.remove_background_noise(beam)

        # remove transmitter channel from beam data
        filtered_beam = Filters.remove_transmitter_channel(filtered_beam)

        # detect debris track using hough transform
        candidates = sdd.detect(beam = filtered_beam)

        # extract needed parameters and dump to file / web service
        # todo encapsulate this logic in a separate class (model?)
        if self.persist_results:
            filtered_beam.save(file_name = 'filtered_beam')
            candidates.view_candidates(output_dir = self.beams_output_dir, beam=beam)
            candidates.save_candidates(output_dir = self.beams_output_dir)

from app.sandbox.postprocessing.lib.BeamDataFilter import Filters
from app.sandbox.postprocessing.lib.SpaceDebrisDetection import *
from app.sandbox.postprocessing.models.Beam import Beam

import os


class SpaceDebrisController:
    def __init__(self):
        # todo parameters should be taken from a separate configuration file
        self.output_dir = os.path.join('public', 'results')
        self.observation_name = 'mock_observation'
        self.beams_output_dir = os.path.join(self.output_dir, self.observation_name, 'beams')
        self.persist_results = True

    def run(self):
        # todo repeat for each beam
        # get input data to consume
        beam = Beam(beam_id = 1, d_delta = 1.0, dha = 1.25)

        if self.persist_results:
            self.save_beam_data(beam, 'input_beam')

        sdd = SpaceDebrisDetection(LineSpaceDebrisDetectionStrategy(max_detections = 3))

        # remove background noise from beam data
        filtered_beam = Filters.remove_background_noise(beam)

        # remove transmitter channel from beam data
        filtered_beam = Filters.remove_transmitter_channel(filtered_beam)

        # detect debris track using hough transform
        detections = sdd.detect(beam = filtered_beam)

        # extract needed parameters and dump to file / web service
        # todo encapsulate this logic in a separate class (model?)
        if self.persist_results:
            self.save_beam_data(filtered_beam, 'filtered_beam')
            self.save_candidates_data(detections = detections)

    # todo save to database instead of FS (inside a model)
    def save_candidates_data(self, detections):
        for i, detection in enumerate(detections):
            beam_id = 'beam_' + str(detection.beam.id)
            candidate_id = 'candidate_' + str(i)
            file_path = os.path.join(self.beams_output_dir, beam_id, 'candidates', candidate_id)

            if not os.path.exists(file_path):
                os.makedirs(file_path)

            detection.save(file_path = os.path.join(file_path, 'orbit_determination_data'),
                           name = candidate_id)  # generate table

            detection.view(file_path = os.path.join(file_path, 'detection_profile'),
                           name = candidate_id)  # generate heat map

        print 'There were', len(detections), 'detections'

    # todo save to database instead of FS (inside a model)
    def save_beam_data(self, beam, file_name):
        beam_id = 'beam_' + str(beam.id)
        file_path = os.path.join(self.beams_output_dir, beam_id, file_name)
        beam.data.view(file_path)
        exit(0)
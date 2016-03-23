from app.sandbox.postprocessing.lib.BeamDataFilter import Filters
from app.sandbox.postprocessing.lib.SpaceDebrisDetection import *
from app.sandbox.postprocessing.models.BeamData import *


class SpaceDebrisController:
    def __init__(self):
        # todo parameters should be taken from a separate configuration file
        return

    def run(self):
        # todo repeat for each beam
        # get input data to consume
        beam = Beam(beam_id = 1, d_delta = 1.0, dha = 1.25)
        beam.data.view()
        sdd = SpaceDebrisDetection(LineSpaceDebrisDetectionStrategy(max_detections = 3))

        # remove background noise from beam data
        filtered_beam_data = Filters.remove_background_noise(beam.data)

        # remove transmitter channel from beam data
        filtered_beam_data = Filters.remove_transmitter_channel(filtered_beam_data)
        filtered_beam_data.view()
        # detect debris track using hough transform
        detections = sdd.detect(beam_id = beam.id, beam_data = filtered_beam_data)

        # extract needed parameters and dump to file / web service
        # todo encapsulate this logic in a separate class
        od_input = self.save_candidates_data(beam_data = filtered_beam_data, detections = detections)

        # return od_input

    @staticmethod
    def save_candidates_data(beam_data, detections):
        print 'There were', len(detections), 'detections'

        for i, detection in enumerate(detections):
            detection.view(beam_data, name = str(i))
            detection.display_parameters()
        parameters = {}
        return parameters

    def visualise_results(self):
        # should run as a separate process so as to not block run
        # visualisation should be a seperate class
        return

from BeamData import *
from SpaceDebrisDetector import SpaceDebrisDetector


class SpaceDebrisController:
    def __init__(self):
        # todo parameters should be taken from a separate configuration file
        return

    def run(self):
        # todo repeat for each beam
        # get input data to consume
        beam = Beam(beam_id = 1, d_delta = 1.0, dha = 1.25)

        sdd = SpaceDebrisDetector(max_detections = 3)

        # remove background noise from beam data
        filtered_beam_data = Filters.remove_background_noise(beam.data)

        # remove transmitter frequency from beam data
        filtered_beam_data = Filters.remove_transmitter_frequency(filtered_beam_data)

        # detect debris track using hough transform
        detections = sdd.get_detections(beam_id = beam.id, beam_data = filtered_beam_data)

        print detections

        # extract needed parameters and dump to file / web service
        # todo encapsulate this logic in a separate class
        od_input = self.get_orbit_determination_input_file(detections)

        # visualise track detected incl. extracted parameters
        # visualise multi-pixel plot

        return

    def get_orbit_determination_input_file(self, detections):
        print 'There were', len(detections), 'detections'
        parameters = {}
        return parameters

    def visualise_results(self):
        # should run as a separate process so as to not block run
        # visualisation should be a seperate class
        return

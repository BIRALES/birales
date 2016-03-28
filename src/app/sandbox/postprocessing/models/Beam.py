from app.sandbox.postprocessing.models.BeamDataMock import BeamDataMock
from app.sandbox.postprocessing.models.BeamDataObservation import BeamDataObservation
import os.path


class Beam:
    def __init__(self, beam_id, d_delta, dha, observation):
        self.id = beam_id
        self.name = 'Beam ' + str(beam_id)
        self.position = None  # todo change to degrees
        self.d_delta = d_delta
        self.dha = dha
        self.observation = observation
        self.output_dir = os.path.join('public', 'results')

        self.data = BeamDataMock(f0 = 0, fn = 200, time = 600)

        # self.data = BeamDataObservation(n_beams = 32, n_channels = 8192, beam = 15)

    def save(self, file_name):
        beams_output_dir = os.path.join(self.output_dir, self.observation, 'beams')
        beam_id = 'beam_' + str(self.id)
        file_path = os.path.join(beams_output_dir, beam_id, file_name)
        self.data.view(file_path)
from app.sandbox.postprocessing.models.BeamDataMock import BeamDataMock
from app.sandbox.postprocessing.models.BeamDataObservation import BeamDataObservation
import app.sandbox.postprocessing.config.application as config

import os.path


class Beam:
    def __init__(self, beam_id, d_delta, dha, observation):

        self.id = beam_id
        self.name = 'Beam ' + str(beam_id)
        self.position = None  # todo change to degrees
        self.d_delta = d_delta
        self.dha = dha

        self.observation = observation

        if config.DATA_SET is 'mock':
            self.data = BeamDataMock(f0 = 0, fn = 200, time = 600)

        else:
            self.data = BeamDataObservation(beam_id = beam_id, observation = observation)

    def save(self, file_name):
        beam_id = 'beam_' + str(self.id)

        file_path = os.path.join(config.BEAM_OUTPUT_DATA, beam_id, file_name)
        self.data.view(file_path)

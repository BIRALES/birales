from app.sandbox.postprocessing.models.BeamDataMock import BeamDataMock
from app.sandbox.postprocessing.models.BeamDataObservation import BeamDataObservation
import app.sandbox.postprocessing.config.application as config

import os.path


class Beam:
    def __init__(self, beam_id, dec, ra, ha, top_frequency, frequency_offset, observation):
        self.id = beam_id
        self.name = 'Beam ' + str(beam_id)

        self.dec = dec
        self.ra = ra
        self.ha = ha
        self.top_frequency = top_frequency
        self.frequency_offset = frequency_offset

        self.observation = observation

        if observation.data_set is 'mock':
            self.data = BeamDataMock(f0 = 0, fn = 200, time = 600)
        else:
            self.data = BeamDataObservation(beam_id = beam_id, observation = observation)

    def save(self, file_name):
        beam_id = 'beam_' + str(self.id)

        file_path = os.path.join(self.observation.beam_output_data, beam_id, file_name)
        self.data.view(file_path)

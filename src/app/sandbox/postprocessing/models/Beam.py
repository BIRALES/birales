from app.sandbox.postprocessing.models.BeamDataMock import BeamDataMock
from app.sandbox.postprocessing.models.BeamDataObservation import BeamDataObservation


class Beam:
    def __init__(self, beam_id, d_delta, dha):
        self.id = beam_id
        self.name = 'Beam ' + str(beam_id)
        self.position = None  # todo change to degrees
        self.d_delta = d_delta
        self.dha = dha

        self.data = BeamDataMock(f0 = 0, fn = 200, time = 600)
        self.data = BeamDataObservation(n_beams = 32, n_channels = 8192, beam = 15)

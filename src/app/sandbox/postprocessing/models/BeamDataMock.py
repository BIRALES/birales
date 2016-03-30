import numpy as np
from app.sandbox.postprocessing.helpers.LineGeneratorHelper import LineGeneratorHelper
from app.sandbox.postprocessing.models.BeamData import BeamData


class BeamDataMock(BeamData):
    def __init__(self, f0, fn, time):
        BeamData.__init__(self)

        # Mock beam data parameters
        self.channels = np.linspace(f0, fn, fn)
        self.time = np.linspace(0, time, time)
        self.snr = None

        self.noise_lvl = 0.1  # The standard deviation of the normal distribution noise
        self.name = 'Mock up of a Beam Data'
        self.set_data()

    def set_data(self):
        """
        Build sample / mock up data to be used for testing
        """
        snr = np.zeros((max(self.channels), self.time[-1]))

        snr = self.add_mock_noise(noiseless_data = snr)
        snr = self.add_mock_track(snr, (50, 100), (100, 200))
        snr = self.add_mock_track(snr, (20, 20), (70, 60))
        # snr = self.add_mock_track(snr, (450, 20), (500, 190))

        self.snr = snr

    def add_mock_noise(self, noiseless_data):
        """
        Add noise to the passed numpy array
        :param noiseless_data:
        :return: noisy data
        """

        noise = abs(np.random.normal(0, self.noise_lvl, size = noiseless_data.shape))

        return noiseless_data + noise

    @staticmethod
    def add_mock_track(snr, start_coordinate = (120, 90), end_coordinate = (180, 120)):
        # todo set limits and throw exception is needed
        track_points = LineGeneratorHelper.get_line(start_coordinate, end_coordinate)  # (x0, y0), (x1, y1)

        for (channel, time) in track_points:
            snr[channel][time] = 1.0

        return snr

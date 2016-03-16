import numpy as np
from abc import abstractmethod
from helpers.visualisation.BeamDataVisualisation import BeamDataVisualisation
from helpers.LineGeneratorHelper import LineGeneratorHelper


class Beam:
    def __init__(self, beam_id, d_delta, dha):
        self.id = beam_id
        self.name = 'Beam ' + str(beam_id)
        self.position = None  # todo change to degrees
        self.d_delta = d_delta
        self.dha = dha

        self.data = BeamDataMock(f0 = 0, fn = 200, time = 600)


class BeamData:
    def __init__(self):
        self.frequencies = None
        self.time = None
        self.snr = None
        self.name = None

    def view(self, name = None):
        if not name:
            name = self.name
        view = BeamDataVisualisation(name)
        view.set_layout(figure_title = name,
                        x_axis_title = 'Frequency (Hz)',
                        y_axis_title = 'Time (s)')

        view.set_data(data = self.snr,
                      x_axis = self.time,
                      y_axis = self.frequencies)

        view.show()

    @abstractmethod
    def set_data(self):
        pass


class BeamDataMedicina(BeamData):
    def __init__(self):
        BeamData.__init__(self)

        self.name = 'Beam Data from Medicina'
        self.set_data()

    def set_data(self):
        return


class BeamDataMock(BeamData):
    def __init__(self, f0, fn, time):
        BeamData.__init__(self)

        # Mock beam data parameters
        self.frequencies = np.linspace(f0, fn, fn)
        self.time = time
        self.snr = None

        self.noise_lvl = 0.2  # The standard deviation of the normal distribution noise
        self.name = 'Mock up of a Beam Data'
        self.set_data()

    def set_data(self):
        """
        Build sample / mock up data to be used for testing
        """
        snr = np.zeros((max(self.frequencies), self.time))
        snr = self.add_mock_noise(noiseless_data = snr)
        snr = self.add_mock_track(snr)

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
    def add_mock_track(snr, start_coordinate = (90, 120), end_coordinate = (400, 180)):
        track_points = LineGeneratorHelper.get_line(start_coordinate, end_coordinate)  # (x0, y0), (x1, y1)

        for (time, frequency) in track_points:
            snr[frequency][time] = 1.0

        return snr


class Filters:
    def __init__(self):
        pass

    @staticmethod
    def remove_background_noise(beam_data):
        # Remove instaces that are 5 stds away from the mean
        data = beam_data.snr
        data[data < np.mean(data) + 5. * np.std(data)] = 0.

        return beam_data

    @staticmethod
    def remove_transmitter_frequency(beam_data):
        # todo remove transmitter frequency
        return beam_data

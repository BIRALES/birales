import numpy as np


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

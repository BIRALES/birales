import numpy as np


class Filters:
    def __init__(self):
        pass

    @staticmethod
    def remove_background_noise(beam):
        # Remove instaces that are 5 stds away from the mean
        beam.data.snr[beam.data.snr < np.mean(beam.data.snr) + 5. * np.std(beam.data.snr)] = 0.

        return beam

    @staticmethod
    def remove_transmitter_channel(beam_data):
        # todo remove transmitter channel
        return beam_data

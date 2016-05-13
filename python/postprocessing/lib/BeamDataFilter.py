import numpy as np


class Filters:

    def __init__(self):
        pass

    @staticmethod
    def remove_background_noise(beam, std = 3.):
        """
        Remove instances that are n stds away from the mean

        :param beam:
        :param std:
        :return:
        """

        beam.snr[beam.snr < np.mean(beam.snr) + std * np.std(beam.snr)] = 0.

        return beam

    @staticmethod
    def remove_transmitter_channel(beam):
        """
        Remove the main transmission beam from the beam data
        :param beam:
        :return:
        """
        sum_across_time = beam.snr.sum(axis = 1)
        # todo determine how 20.0 threshold is determined
        peaks = np.where(sum_across_time > 20.)[0]

        for i, peak in enumerate(peaks):
            peak_snr = beam.snr[peak]
            mean = np.mean(peak_snr[peak_snr > 0.0])
            beam.snr[peak][peak_snr > 0.0] -= mean
            beam.snr[peak][peak_snr < 0.0] = 0.0

        beam = Filters.remove_background_noise(beam)
        return beam

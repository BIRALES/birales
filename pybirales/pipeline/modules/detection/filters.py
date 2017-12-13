import logging as log
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal

from abc import abstractmethod


class BeamDataFilter:
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, beam):
        """
        Apply filter on the beam data
        :param beam:
        :return: void
        """
        pass


class RemoveTransmitterChannelFilter(BeamDataFilter):
    def __init__(self):
        BeamDataFilter.__init__(self)

        # todo - Determine how 20.0 threshold is determined
        self.threshold = 20.

    def apply(self, beam):
        """
        Remove the main transmission beam from the beam data
        :param beam:
        :return: void
        """

        # sum_across_time = beam.snr.sum(axis=1)
        # peaks = np.where(sum_across_time > self.threshold)[0]
        #
        # for i, peak in enumerate(peaks):
        #     peak_snr = beam.snr[peak]
        #     mean = np.mean(peak_snr[peak_snr > 0.0])
        #     beam.snr[peak][peak_snr > 0.0] -= mean
        #     beam.snr[peak][peak_snr < 0.0] = 0.0

        sum_across_time = beam.snr.sum(axis=0)
        # peaks = np.where(sum_across_time > self.threshold)[0]

        peaks_snr_i = np.where(sum_across_time > self.threshold)
        beam.snr[:, peaks_snr_i] = beam.snr[:, peaks_snr_i] - np.mean(beam.snr[:, peaks_snr_i], axis=0)
        beam.snr[beam.snr < 0.0] = 0.



        # for i, peak in enumerate(peaks):
        #     peak_snr = beam.snr[peak]
        #     mean = np.mean(peak_snr[peak_snr > 0.0])
        #     beam.snr[peak][peak_snr > 0.0] -= mean
        #     beam.snr[peak][peak_snr < 0.0] = 0.0

        log.debug('Filter: Transmitter frequency removed from filtered beam %s', beam.id)


class RemoveBackgroundNoiseFilter(BeamDataFilter):
    def __init__(self, std_threshold):
        BeamDataFilter.__init__(self)
        self.std_threshold = std_threshold

    def apply(self, beam):
        """
        Remove instances that are n std away from the mean
        :param beam:
        :return: void
        """

        mean = np.mean(beam.snr)
        std = np.std(beam.snr)
        threshold = self.std_threshold * std + mean

        try:
            beam.snr[beam.snr < threshold] = 0.
        except Exception:
            print(beam.snr)
        log.debug('Beam %s: Background noise removed', beam.id)


class PepperNoiseFilter(BeamDataFilter):
    def __init__(self):
        BeamDataFilter.__init__(self)

    def apply(self, beam):
        structure = np.zeros((5, 5))
        structure[2, 2] = 1

        hot_pixels_mask = ndimage.binary_hit_or_miss(beam.snr, structure1=structure)
        beam.snr[hot_pixels_mask] = 0.


class MedianFilter(BeamDataFilter):
    def __init__(self):
        BeamDataFilter.__init__(self)

    def apply(self, beam):
        """
        Apply a median filter on the beam data

        :param beam:
        :return: void
        """

        # beam.snr = signal.medfilt(beam.snr, 3)

        # from scipy.ndimage import median_filter
        # beam.snr = median_filter(beam.snr, size=2) - beam.snr
        log.debug('Beam %s: Median filter applied', beam.id)


class InputDataFilter:
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, input_data):
        """
        Apply filter on the input data
        :param input_data:
        :return: void
        """
        pass


class TransmitterInputDataFilter(InputDataFilter):
    def __init__(self):
        InputDataFilter.__init__(self)

        # todo - Determine how 20.0 threshold is determined
        self.threshold = 20.

    def apply(self, beam):
        """
        Remove the main transmission beam from the beam data
        :param beam:
        :return: void
        """

        sum_across_time = beam.snr.sum(axis=1)
        peaks = np.where(sum_across_time > self.threshold)[0]

        for i, peak in enumerate(peaks):
            peak_snr = beam.snr[peak]
            mean = np.mean(peak_snr[peak_snr > 0.0])
            beam.snr[peak][peak_snr > 0.0] -= mean
            beam.snr[peak][peak_snr < 0.0] = 0.0

        log.debug('Beam %s: Transmitter frequency removed', beam.id)

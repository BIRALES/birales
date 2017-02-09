import logging as log
import numpy as np
import scipy.ndimage as ndimage

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

        sum_across_time = beam.snr.sum(axis=1)
        peaks = np.where(sum_across_time > self.threshold)[0]

        for i, peak in enumerate(peaks):
            peak_snr = beam.snr[peak]
            mean = np.mean(peak_snr[peak_snr > 0.0])
            beam.snr[peak][peak_snr > 0.0] -= mean
            beam.snr[peak][peak_snr < 0.0] = 0.0

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
        beam.snr[beam.snr < threshold] = 0.

        log.debug('Filter: Background noise removed from input beam %s', beam.id)


class MedianFilter(BeamDataFilter):
    def __init__(self):
        BeamDataFilter.__init__(self)

    def apply(self, beam):
        """
        Apply a median filter on the beam data

        :param beam:
        :return: void
        """

        beam.snr = ndimage.median_filter(beam.snr, 1)

        log.debug('Filter: Median filter applied on beam %s', beam.id)

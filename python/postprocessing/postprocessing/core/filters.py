import numpy as np
from abc import abstractmethod
import logging as log


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

        log.debug('Transmitter frequency removed from filtered beam %s', beam.id)


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

        beam.snr[beam.snr < np.mean(beam.snr) + self.std_threshold * np.std(beam.snr)] = 0.

        log.debug('Background noise removed from input beam %s', beam.id)

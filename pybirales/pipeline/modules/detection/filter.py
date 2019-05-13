import logging as log
from abc import abstractmethod

import numpy as np
from scipy.ndimage import binary_hit_or_miss

from pybirales import settings
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob


class InputDataFilter:
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, input_data, obs_info):
        """
        Apply filter on the beam data
        :param input_data:
        :param obs_info:
        :return: void
        """
        pass


class RemoveBackgroundNoiseFilter(InputDataFilter):
    def __init__(self, std_threshold):
        InputDataFilter.__init__(self)
        self.std_threshold = std_threshold

    def apply(self, data, obs_info):
        """
        Remove instances that are n std away from the mean
        :param data:
        :param obs_info:
        :return: void
        """

        # mean = np.mean(data, axis=(1, 2), keepdims=True)
        # std = np.std(data, axis=(1, 2), keepdims=True)
        # threshold = self.std_threshold * std + mean

        # Calculate the threshold at which the noise will be clipped
        t2 = 3 * obs_info['channel_noise_std'] + obs_info['channel_noise']

        log.debug('Noise: {:0.2f}W, Threshold set at {:0.2f}W'.format(np.mean(obs_info['channel_noise']), np.mean(t2)))
        # re-shape threshold array so to make it compatible with the data

        # print np.shape(t2), data.shape
        t2 = np.expand_dims(t2, axis=2)

        # print np.shape(t2), data.shape
        data[data <= t2] = -100.

        # print np.min(data)


class PepperNoiseFilter(InputDataFilter):
    def __init__(self):
        InputDataFilter.__init__(self)
        self._structure = np.zeros((3, 3))
        self._structure[1, 1] = 1

    def _remove_pepper_noise(self, data):
        return binary_hit_or_miss(data, structure1=self._structure)

    def apply(self, data, obs_info):
        """
        Remove speck noise (salt-and-pepper)
        :param data:
        :param obs_info:
        :return: void
        """
        # todo - can this for loop be eliminated?
        for beam_id in range(data.shape[0]):
            data[beam_id, self._remove_pepper_noise(data[beam_id])] = -100


class RemoveTransmitterChannelFilter(InputDataFilter):
    def __init__(self):
        InputDataFilter.__init__(self)

        # todo - Determine how 20.0 threshold is determined
        self.threshold = 20.

    def apply(self, data, obs_info):
        """
        Remove the main transmission beam from the beam data
        :param data:
        :return: void
        """

        # peaks_snr_i = np.where(np.sum(data, axis=2) > self.threshold)
        # data[peaks_snr_i] = data[peaks_snr_i] - np.mean(data[peaks_snr_i], axis=1, keepdims=True)
        # data[data < 0.0] = 0.

        if settings.detection.filter_transmitter:
            summed = np.sum(data, axis=2)
            peaks_snr_i = np.unique(np.where(summed > np.mean(summed) + np.std(summed) * 5.0)[1])
            data[:, peaks_snr_i, :] = -100
            log.debug('Transmitter frequency filter applied')


class Filter(ProcessingModule):
    _valid_input_blobs = [ChannelisedBlob]

    def __init__(self, config, input_blob=None):
        # Ensure that the input blob is of the expected format
        self._validate_data_blob(input_blob, valid_blobs=[ChannelisedBlob])


        # The filters to be applied on the data. Filters will be applied in order.
        self._filters = [
            RemoveBackgroundNoiseFilter(std_threshold=4.),
            # RemoveTransmitterChannelFilter(),
            # PepperNoiseFilter(),
        ]

        super(Filter, self).__init__(config, input_blob)

        self.name = "Filtering"

    def process(self, obs_info, input_data, output_data):
        """
        Filter the channelised data

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # Skip the first blob
        if self._iter_count < 1:
            return

        # Apply the filters on the data
        for f in self._filters:
            f.apply(input_data, obs_info)

        output_data[:] = input_data

        return obs_info

    def generate_output_blob(self):
        # Generate output blob
        return ChannelisedBlob(self._config, self._input.shape, datatype=self._input.datatype)

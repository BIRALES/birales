import logging as log
from abc import abstractmethod

import numpy as np
from scipy.ndimage import binary_hit_or_miss
from skimage.filters import threshold_triangle

from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob


def triangle_filter(input_data, obs_info):
    for b in range(0, input_data.shape[0]):
        beam_data = input_data[b, ...]
        local_thresh = threshold_triangle(beam_data, nbins=1024)
        local_filter_mask = beam_data < local_thresh

        beam_data[local_filter_mask] = -100

    return input_data


def sigma_clip(input_data, obs_info):
    threshold = 3 * obs_info['channel_noise_std'] + obs_info['channel_noise']
    t2 = np.expand_dims(threshold, axis=2)
    input_data[input_data <= t2] = 0

    return input_data


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

        t2 = self.std_threshold * obs_info['channel_noise_std'] + obs_info['channel_noise']

        log.debug('Noise: {:0.2f}W, Threshold set at {:0.2f}W'.format(np.mean(obs_info['channel_noise']), np.mean(t2)))
        # re-shape threshold array so to make it compatible with the data

        # print np.shape(t2), data.shape
        t2 = np.expand_dims(t2, axis=2)

        # print np.shape(t2), data.shape
        data[data <= t2] = -50.

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


class RemoveTransmitterChannelFilterPeak(InputDataFilter):
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

        summed = np.sum(data, axis=2)
        peaks_snr_i = np.unique(np.where(summed > np.mean(summed) + np.std(summed) * 5.0)[1])
        data[:, peaks_snr_i, :] = -100


class RemoveTransmitterChannelFilter(InputDataFilter):
    def __init__(self):
        InputDataFilter.__init__(self)

        self._n_rfi_samples_thold = 0.6  # ~ 90 samples

    def apply(self, data, obs_info):
        """
        Remove the main transmission beam from the beam data
        :param data:
        :return: void
        """

        beam_noise_est = np.mean(obs_info['channel_noise'], axis=1)
        bs = np.sum(data, axis=2)
        mean_bs = np.mean(bs, axis=1)
        std_bs = np.std(bs, axis=1)

        mean_bsr = np.repeat(mean_bs[:, np.newaxis], data.shape[1], axis=1)
        std_bsr = np.repeat(std_bs[:, np.newaxis], data.shape[1], axis=1)

        ndx = np.column_stack(np.where(bs > mean_bsr + 3 * std_bsr))

        for b in range(0, data.shape[0]):
            rfi_channels = ndx[ndx[:, 0] == b, 1]

            # need to check for consistency
            beam_noise = beam_noise_est[b]
            for c in rfi_channels:
                channel_data = data[b, c, :]
                n2 = len(channel_data[channel_data > beam_noise])

                # Channels which consistently have a high power (more than mean) are masked ~ 90 samples
                if n2 > self._n_rfi_samples_thold * data.shape[2]:
                    data[b, c, :] = beam_noise


class TriangleFilter(InputDataFilter):
    def __init__(self):
        InputDataFilter.__init__(self)

    def apply(self, data, obs_info):
        triangle_filter(data, obs_info)


class SigmaClipFilter(InputDataFilter):
    def __init__(self):
        InputDataFilter.__init__(self)

    def apply(self, data, obs_info):
        sigma_clip(data, obs_info)


class Filter(ProcessingModule):
    _valid_input_blobs = [ChannelisedBlob]

    def __init__(self, config, input_blob=None):
        # Ensure that the input blob is of the expected format
        self._validate_data_blob(input_blob, valid_blobs=[ChannelisedBlob])

        # The filters to be applied on the data. Filters will be applied in order.

        # self._filters = [
        #     RemoveTransmitterChannelFilter(),
        #     RemoveBackgroundNoiseFilter(std_threshold=4.),
        #     PepperNoiseFilter(),
        # ]

        # # For DBSCAN
        # self._filters = [
        #     RemoveTransmitterChannelFilter(),
        #     SigmaClipFilter(),
        #     PepperNoiseFilter(),
        # ]

        # For MSDS
        self._filters = [
            RemoveTransmitterChannelFilter(),
            TriangleFilter(),
            PepperNoiseFilter(),
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
        if self._iter_count < 0:
            return

        # print '\nbefore filter', self._iter_count, input_data[input_data > 0].shape
        # Apply the filters on the data
        for f in self._filters:
            f.apply(input_data, obs_info)

        output_data[:] = input_data

        # print '\nafter filter', self._iter_count, input_data[input_data > 0].shape

        return obs_info

    def generate_output_blob(self):
        # Generate output blob
        return ChannelisedBlob(self._config, self._input.shape, datatype=self._input.datatype)

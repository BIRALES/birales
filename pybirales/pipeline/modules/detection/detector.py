import logging as log
from multiprocessing import Pool

import numpy as np
from scipy import stats

from pybirales import settings

from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.pipeline.modules.detection.dbscan_detection import detect
from pybirales.pipeline.modules.detection.exceptions import DetectionClusterIsNotValid
from pybirales.pipeline.modules.detection.space_debris_candidate import SpaceDebrisTrack
from pybirales.pipeline.base.timing import timeit
from functools import partial


class Detector(ProcessingModule):
    _valid_input_blobs = [ChannelisedBlob]

    def __init__(self, config, input_blob=None):
        # Ensure that the input blob is of the expected format
        self._validate_data_blob(input_blob, valid_blobs=[ChannelisedBlob])

        if settings.detection.multi_proc:
            self.pool = Pool(4)

        self.channels = None

        self._doppler_mask = None

        super(Detector, self).__init__(config, input_blob)

        self.name = "Detector"

        # A list of space debris tracks
        self._candidates = []

    def _tear_down(self):
        """

        :return:
        """
        if settings.detection.multi_proc:
            self.pool.close()

    def _apply_doppler_mask(self, obs_info):
        """

        :param obs_info:
        :return:
        """
        if self._doppler_mask is None:
            self.channels = np.arange(obs_info['start_center_frequency'],
                                      obs_info['start_center_frequency'] + obs_info['channel_bandwidth'] * obs_info[
                                          'nchans'], obs_info['channel_bandwidth'])
            a = obs_info['transmitter_frequency'] + settings.detection.doppler_range[0] * 1e-6
            b = obs_info['transmitter_frequency'] + settings.detection.doppler_range[1] * 1e-6

            self._doppler_mask = np.bitwise_and(self.channels < b, self.channels > a)

            if settings.detection.enable_doppler_window:
                self.channels = self.channels[self._doppler_mask]
            else:
                # Select all channels
                self._doppler_mask = False

        return self.channels, self._doppler_mask

    def _debug_msg(self, cluster):
        """

        :param cluster:
        :return:
        """
        m, c, r_value, _, _ = stats.linregress(cluster['channel_sample'], cluster['time_sample'])
        return '{} (m={:0.2f}, c={:0.2f}, s={:0.2f}, n={}, i={})'.format(id(cluster), m, c, r_value, cluster.shape[0],
                                                                         self._iter_count)

    @timeit
    def _aggregate_clusters(self, candidates, clusters, obs_info):
        """
        Create Space Debris Tracks from the detection clusters identified in each beam

        :param clusters:
        :return:
        """

        for cluster in clusters:
            for candidate in self._candidates:
                # If beam candidate is similar to candidate, merge it.
                if candidate.is_parent_of(cluster):
                    try:
                        candidate.add(cluster)
                    except DetectionClusterIsNotValid:
                        log.debug('Beam candidate {} could not be added to track {}'.format(
                            self._debug_msg(cluster), id(candidate)))
                    else:
                        log.debug('Beam candidate {} added to track {}'.format(self._debug_msg(cluster),
                                                                               id(candidate)))

                        # print 'candidate is now {} long'.format(candidate.size)
                        break
            else:
                # Beam cluster does not match any candidate. Create a new candidate track from it.
                try:
                    sd = SpaceDebrisTrack(obs_info=obs_info, cluster=cluster)
                except DetectionClusterIsNotValid:
                    continue

                log.debug('Created new track {} from Beam candidate {}'.format(id(sd), self._debug_msg(cluster)))

                # Add the space debris track to the candidates list
                candidates.append(sd)

        # Notify the listeners, that a new detection was made
        if settings.observation.notifications:
            [candidate.send_notification() for candidate in self._candidates]

        # Save candidates that were updated
        if settings.detection.save_candidates:
            [candidate.save() for candidate in self._candidates if not candidate.saved]

        return candidates

    @timeit
    def _active_tracks(self, candidates, iter_count):
        """

        :param candidates:
        :param iter_count:
        :return:
        """

        temp_candidates = []
        for candidate in candidates:
            if candidate.has_transitted(iter_count=iter_count):
                # If the candidate is not valid delete it
                if not candidate.is_valid:
                    candidate.delete()
            else:
                temp_candidates.append(candidate)

        log.info('Result: {} tracks have transitted. {} tracks are currently in detection window.'.format(
            len(self._candidates) - len(temp_candidates), len(temp_candidates)))

        return temp_candidates

    def _pre_process(self, input_data, obs_info):
        """

        :param input_data:
        :param obs_info:
        :return:
        """
        channels, doppler_mask = self._apply_doppler_mask(obs_info)
        channel_noise = obs_info['channel_noise']

        if settings.detection.enable_doppler_window:
            input_data = input_data[:, doppler_mask, :]
            channel_noise = channel_noise[:, doppler_mask]

        log.info('Using {} out of {} channels ({:0.3f} MHz to {:0.3f} MHz).'.format(channel_noise.shape[1],
                                                                                    obs_info['nchans'],
                                                                                    channels[0],
                                                                                    channels[-1]
                                                                                    ))

        return input_data, channels, channel_noise

    @timeit
    def _get_clusters(self, input_data, channels, channel_noise, obs_info, iter_counter):
        """

        :param input_data:
        :param channels:
        :param channel_noise:
        :param obs_info:
        :param iter_counter:
        :return:
        """

        t0 = np.datetime64(obs_info['timestamp'])
        td = np.timedelta64(int(obs_info['sampling_time'] * 1e9), 'ns')

        # todo - multi-processing is slower than single threaded equivalent
        if settings.detection.multi_proc:
            func = partial(detect, input_data, channels, t0, td, iter_counter, channel_noise)
            clusters = self.pool.map(func, range(0, 32))
            return [val for sublist in clusters for val in sublist]
        else:
            clusters = []
            for beam_id in range(0, 32):
                clusters.extend(detect(input_data, channels, t0, td, iter_counter, channel_noise, beam_id))
            return clusters

    def process(self, obs_info, input_data, output_data):
        """
        Sieve through the pre-processed channelised data for space debris candidates

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # Skip the first blob
        if self._iter_count < 1:
            return

        obs_info['iter_count'] = self._iter_count

        # Pre-process the input data
        input_data, channels, channel_noise = self._pre_process(input_data, obs_info)

        # Process the input data and identify the detection clusters
        clusters = self._get_clusters(input_data, channels, channel_noise, obs_info, self._iter_count)

        # Create new tracks from clusters or merge clusters into existing tracks
        candidates = self._aggregate_clusters(self._candidates, clusters, obs_info)

        # Check each track and determine if the detection object has transitted outside FoV
        self._candidates = self._active_tracks(candidates, self._iter_count)

        return obs_info

    def generate_output_blob(self):
        pass

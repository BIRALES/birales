import logging as log
import time
from multiprocessing import Pool

from scipy import stats

from pybirales import settings
from pybirales.events.events import TrackTransittedEvent
from pybirales.events.publisher import publish
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.pipeline.modules.detection.dbscan_detection import detect
from pybirales.pipeline.modules.detection.exceptions import DetectionClusterIsNotValid
from pybirales.pipeline.modules.detection.filter import RemoveBackgroundNoiseFilter, RemoveTransmitterChannelFilter
from pybirales.pipeline.modules.detection.msds.msds import *
from pybirales.pipeline.modules.detection.msds.util import _create_cluster
from pybirales.pipeline.modules.detection.msds.visualisation import *
from pybirales.pipeline.modules.detection.space_debris_candidate import SpaceDebrisTrack
import pandas as pd


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

        # A list of space debris tracks currently in memory
        self._candidates = []

        # Number of RSO tracks detected in this observation
        self._n_rso = 0

        self.filter_tx = RemoveTransmitterChannelFilter()
        self.filter_bkg3 = RemoveBackgroundNoiseFilter(std_threshold=3)
        self.filter_bkg4 = RemoveBackgroundNoiseFilter(std_threshold=4)

        self.detections = {}

    def _tear_down(self):
        """

        :return:
        """
        # if settings.detection.multi_proc:
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

            self.channels = self.channels[self._doppler_mask]

        return self.channels, self._doppler_mask

    def _debug_msg(self, cluster):
        """

        :param cluster:
        :return:
        """
        m, c, r_value, _, _ = stats.linregress(cluster['channel_sample'], cluster['time_sample'])
        return '{:03} (m={:0.2f}, c={:0.2f}, s={:0.2f}, n={}, i={})'.format(id(cluster) % 100, m, c, r_value,
                                                                            cluster.shape[0],
                                                                            self._iter_count)

    # @timeit
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
                        log.debug('Beam candidate {} could not be added to track {:03}'.format(
                            self._debug_msg(cluster), id(candidate) % 1000))
                    else:
                        log.debug('Beam candidate {} added to track {:03}'.format(self._debug_msg(cluster),
                                                                                  id(candidate) % 1000))

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

    # @timeit
    def _active_tracks(self, candidates, iter_count):
        """

        :param candidates:
        :param iter_count:
        :return:
        """

        temp_candidates = []
        # Tracks that were deleted should not count as 'track transitted'
        invalid_tracks = 0
        transitted = 0
        for candidate in candidates:
            if not candidate.is_valid():
                candidate.delete()

                invalid_tracks += 1
                continue

            if candidate.has_transitted(iter_count=iter_count):
                # If the candidate is not valid delete it else it won't be added to the list
                # if not candidate.is_valid:
                #     candidate.delete()
                # else:
                # Track has transitted outside the field of view of the instrument
                publish(TrackTransittedEvent(candidate))
                transitted += 1
            else:
                temp_candidates.append(candidate)

        # transitted = len(self._candidates) - len(temp_candidates) - invalid_tracks
        self._n_rso += transitted
        log.info('Result: {} tracks have transitted. {} tracks are currently in detection window.'.format(transitted,
                                                                                                          len(
                                                                                                              temp_candidates)))

        return temp_candidates

    def _msds_get_merged_input(self, input_data):
        merged_input = np.sum(input_data, axis=0)

        # power
        merged_input = np.power(np.abs(merged_input), 2.0) + 0.00000000000001

        return merged_input

    def _msds_get_merged_filtered_input(self, input_data, obs_info):
        # filter before
        input_data = np.power(np.abs(input_data), 2.0) + 0.00000000000001
        # print '1', obs_info['iter_count'], input_data.min(), input_data.max(), input_data.mean()
        # self.filter_bkg3.apply(input_data, obs_info)
        # input_data[input_data < 0] = 0

        threshold = 3 * np.mean(input_data, axis=2) + np.mean(input_data, axis=2)
        t2 = np.expand_dims(threshold, axis=2)
        input_data[input_data <= t2] = 0

        # print '2', obs_info['iter_count'], input_data.min(), input_data.max(), input_data.mean()
        input_data[input_data < 0] = 0
        # print '3', obs_info['iter_count'], input_data.min(), input_data.max(), input_data.mean()

        merged_input = np.sum(input_data, axis=0)

        # power
        # merged_input = np.power(np.abs(merged_input), 2.0) + 0.00000000000001

        obs_info['channel_noise'] = np.mean(merged_input, axis=1)
        obs_info['channel_noise_std'] = np.std(merged_input, axis=1)


        threshold = 2 * obs_info['channel_noise_std'] + obs_info['channel_noise']
        t2 = np.expand_dims(threshold, axis=1)
        merged_input[merged_input <= t2] = 0


        return merged_input, obs_info

    def _msds_get_naive_input(self, input_data, obs_info):
        obs_info['channel_noise'], obs_info['channel_noise_std'] = np.mean(input_data, axis=2), np.std(input_data,
                                                                                                       axis=2)
        self.filter_bkg3.apply(input_data, obs_info)
        input_data[input_data < 0] = 0

        return input_data, obs_info

    # @timeit
    def _get_clusters_msds(self, input_data, channels, channel_noise, obs_info, iter_counter):
        """
        Multi-pixel streak detection strategy

        Algorithm combines the beam data across the multi-pixel in order to increase SNR whilst
        reducing the computational load. Data is transformed such that data points belonging
        to the same streak, cluster around a common point.  Then, a noise-aware clustering algorithm,
        such as DBSCAN, can be applied on the data points to identify the candidate tracks.

        :param input_data:
        :param channels:
        :param channel_noise:
        :param obs_info:
        :param iter_counter:
        :return:
        """

        debug = self._iter_count in [4, 5, 6]

        input_data_copy = input_data.copy()

        # naive_input, obs_info = self._msds_get_naive_input(input_data, obs_info)

        noise_estimate = np.power(np.abs(np.mean(channel_noise)), 2.0) + 0.00000000000001

        merged_input = self._msds_get_merged_input(input_data_copy.copy())

        filtered_merged_input, obs_info = self._msds_get_merged_filtered_input(input_data_copy, obs_info)

        visualise_input_data(merged_input, input_data, filtered_merged_input, 15, self._iter_count, debug)

        limits = (0, 160, 1600, 2200)
        true_tracks = None
        # input_data_merged = input_data[15, :, :]

        ndx = pre_process_data(filtered_merged_input, noise_estimate=noise_estimate)

        if ndx.shape[0] == 0:
            print 'ndx is empty'
            return []

        # Build quad/nd tree that spans all the data points
        k_tree = build_tree(ndx, leave_size=30, n_axis=2)

        # Traverse the tree and identify valid linear streaks

        # print ndx[:, 0].max(), ndx[:, 1].max(), ndx[:, 0].min(), ndx[:, 1].min()
        leaves = traverse(k_tree.tree, ndx,
                          bbox=(0, ndx[:, 1].max(), 0, ndx[:, 0].max()),
                          distance_thold=3., min_length=2., cluster_size_thold=10.)

        clusters = process_leaves(self.pool, leaves)

        if len(clusters) == 0:
            log.debug("MSDS found no valid linear clusters from {} leaves".format(len(leaves)))
            return []

        visualise_tree_traversal(ndx, true_tracks, clusters, leaves,
                                 '2_processed_leaves_{}.png'.format(self._iter_count), limits=limits, vis=debug)

        # Cluster the leaves based on their vicinity to each other
        eps = estimate_leave_eps(clusters)

        cluster_data, unique_labels = cluster_leaves(clusters, distance_thold=eps)
        log.debug(
            "MSDS identified {} leaves, of which, {} were identified as linear clusters. EPS:{}".format(len(leaves),
                                                                                                        len(clusters),
                                                                                                        eps))

        visualise_clusters(cluster_data, cluster_data[:, 5], unique_labels, true_tracks,
                           filename='3_clusters{}.png'.format(self._iter_count), limits=limits, debug=debug)

        # # Filter invalid clusters
        tracks = validate_clusters(self.pool, cluster_data, unique_labels=unique_labels)

        visualise_tracks(tracks, true_tracks, '5_tracks_{}.png'.format(self._iter_count), limits=limits, debug=debug)

        # Fill any missing data (increase recall)
        valid_tracks = [fill(filtered_merged_input, t) for t in tracks]

        visualise_post_processed_tracks(valid_tracks, true_tracks,
                                        '6_post-processed-tracks_{}.png'.format(self._iter_count), limits=None,
                                        debug=debug)

        # differentiate here.
        return [self.diff(input_data, c, channels, channel_noise, obs_info, iter_counter) for c in valid_tracks]

    def diff(self, input_data, cluster, channels, beam_noise, obs_info, iter_counter):
        chl = cluster[:, 0].astype(int)
        sample = cluster[:, 1].astype(int)
        clusters = []

        power = np.power(np.abs(input_data[:, chl, sample]), 2.0)
        for beam_id in range(0, 32):
            beam_power = power[beam_id]
            noise_estimate = np.mean(beam_power)

            t = np.max(beam_power) - np.std(beam_power)
            # Select only the datapoints who's power is greater than the noise
            cluster_power = beam_power[beam_power > t]

            c = cluster[beam_power > t]
            c[:, 2] = 10 * np.log10(cluster_power / noise_estimate)

            clusters.append(_create_cluster(c, channels, obs_info, beam_id, iter_counter))

        m = pd.concat(clusters)
        print 'Cluster was {} long and this became {} after splitting into the beams'.format(len(cluster), len(m))
        return m

    # @timeit
    def _get_clusters_naive(self, input_data, channels, channel_noise, obs_info, iter_counter):
        """
        Naive algorithm.

        Algorithm uses the DBSCAN algorithm to identify clusters. It iterates over all the beams and considers
        all the pixels that have a non-zero


        :param input_data:
        :param channels:
        :param channel_noise:
        :param obs_info:
        :param iter_counter:
        :return:
        """

        input_data = np.power(np.abs(input_data), 2.0) + 0.00000000000001

        obs_info['channel_noise'] = np.mean(input_data, axis=2)
        obs_info['channel_noise_std'] = np.std(input_data, axis=2)

        threshold = 4 * obs_info['channel_noise_std'] + obs_info['channel_noise']
        t2 = np.expand_dims(threshold, axis=2)
        input_data[input_data <= t2] = 0

        t0 = np.datetime64(obs_info['timestamp'])
        td = np.timedelta64(int(obs_info['sampling_time'] * 1e9), 'ns')

        clusters = []
        for beam_id in range(0, 32):
            # beam_id = 15
            clusters.extend(detect(input_data, channels, t0, td, iter_counter, channel_noise, beam_id))
        log.debug('Found {} new candidates in {} beams'.format(len(clusters), 32))

        return clusters

    def process(self, obs_info, input_data, output_data):
        """
        Sieve through the pre-processed channelised data for space debris candidates

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # print np.shape(input_data)
        # input_data = input_data[0, :, :, :]
        obs_info['iter_count'] = self._iter_count
        obs_info['transitted_tracks'] = []
        channels, doppler_mask = self._apply_doppler_mask(obs_info)

        # Skip the first few blobs (to allow for an accurate noise estimation to be determined)
        if self._iter_count < 2:
            return obs_info

        self.filter_tx.apply(input_data, obs_info)

        input_data = input_data[:, doppler_mask, :]

        # mean of the doppler mask
        obs_info['mean_noise'] = np.mean(obs_info['channel_noise'][:, doppler_mask])
        obs_info['doppler_mask'] = doppler_mask

        channel_noise = obs_info['channel_noise'][:, obs_info['doppler_mask']]

        # print 'channels', channels.shape, input_data.shape
        # [Feature Extraction] Process the input data and identify the detection clusters
        input_data_org = input_data.copy()

        clusters_dbscan = self._get_clusters_naive(input_data_org, channels, channel_noise, obs_info, self._iter_count)
        clusters = clusters_dbscan

        clusters_msds = self._get_clusters_msds(input_data, channels, channel_noise, obs_info, self._iter_count)
        clusters = clusters_msds

        debug = self._iter_count in [4, 5, 6]
        compare_algorithms(input_data_org[15, :, :], msds_clusters=clusters_msds, db_scan_clusters=clusters_dbscan,
                           limits=(0, 160, 1800, 2100), iteration=self._iter_count, debug=debug)

        # [Track Association] Create new tracks from clusters or merge clusters into existing tracks
        candidates = self._aggregate_clusters(self._candidates, clusters, obs_info)

        # [Track Termination] Check each track and determine if the detection object has transitted outside FoV
        self._candidates = self._active_tracks(candidates, self._iter_count)

        # Output a TDM for the tracks that have transitted outside the telescope's FoV
        obs_info['transitted_tracks'] = [c for c in candidates if c not in self._candidates]

        for i, candidate in enumerate(self._candidates):
            log.info("RSO %d: %s", self._n_rso + 1 + i, candidate.state_str())

        return obs_info

    def generate_output_blob(self):
        return ChannelisedBlob(self._config, self._input.shape, datatype=np.float)

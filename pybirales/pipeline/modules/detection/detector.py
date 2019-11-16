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
from msds.msds import *
from msds.util import _create_cluster, snr_calc
from msds.visualisation import *
from pybirales.pipeline.modules.detection.space_debris_candidate import SpaceDebrisTrack
import pandas as pd
import datetime


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

            # doppler_mask = np.ones(shape=8192).astype(bool)
            # return self.channels, doppler_mask

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

    def _msds_get_filtered_input(self, input_data, obs_info, beam_id):
        # filter before
        input_data = np.power(np.abs(input_data[beam_id, ...]), 2.0) + 0.00000000000001

        obs_info['channel_noise'] = np.mean(input_data, axis=1)
        obs_info['channel_noise_std'] = np.std(input_data, axis=1)

        threshold = 3 * np.mean(input_data, axis=1) + np.mean(input_data, axis=1)
        t2 = np.expand_dims(threshold, axis=1)
        input_data[input_data <= t2] = 0

        return input_data, obs_info

    def _msds_get_naive_input(self, input_data, obs_info):
        input_data = np.power(np.abs(input_data), 2.0) + 0.00000000000001

        obs_info['channel_noise'], obs_info['channel_noise_std'] = np.mean(input_data, axis=2), np.std(input_data,
                                                                                                       axis=2)
        self.filter_bkg3.apply(input_data, obs_info)
        input_data[input_data < 0] = 0

        return input_data, obs_info

    def _msds_get_filtered_input2(self, input_data, obs_info):
        input_data = np.power(np.abs(input_data), 2.0) + 0.00000000000001

        obs_info['beam_noise'] = np.mean(np.mean(input_data, axis=1), axis=1)

        input_data_max = np.max(input_data, axis=(1, 2), keepdims=True)
        input_data_min = np.min(input_data, axis=(1, 2), keepdims=True)

        input_data_norm = (input_data - input_data_min) / (input_data_max - input_data_min)

        threshold = 3 * np.std(input_data_norm, axis=2) + np.mean(input_data_norm, axis=2)

        t2 = np.expand_dims(threshold, axis=2)
        input_data_norm[input_data_norm <= t2] = 0

        input_data = input_data_norm

        return input_data, obs_info

    # @timeit
    def _get_clusters_msds(self, input_data, channels, channel_noise, obs_info, iter_counter, limits, debug):
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

        # Filter input data and get the noise_estimate across N beams
        seg_input_data, obs_info = self._msds_get_filtered_input2(input_data, obs_info)

        t1 = time.time()
        clusters = []
        # for b in [14, 20, 21, 22, 23]:
        for b in range(0, 32):
            beam_seg_input_data = seg_input_data[b, ...]

            # todo Get ndx for all beams. Do not do SNR calculation for now.
            # print "noise estimate in beam {} is {} {}".format(b, obs_info['beam_noise'][b],
            #                                                   np.mean(beam_seg_input_data))
            ndx = pre_process_data(beam_seg_input_data, noise_estimate=None)

            # visualise_input_data(None, seg_input_data, None, b, self._iter_count, debug, limits)

            # Create tree on the merged input data
            k_tree = build_tree(ndx, leave_size=40, n_axis=2)

            # Traverse the tree and identify valid linear streaks
            leaves = traverse(k_tree.tree, ndx, bbox=(0, beam_seg_input_data.shape[1], 0, beam_seg_input_data.shape[0]),
                              distance_thold=3., min_length=2., cluster_size_thold=10.)

            positives, negatives = process_leaves(self.pool, leaves, parallel=True)

            if 14 <= b <= 15 and self._iter_count == 4:
                print 'Found {} clusters in beam {} iteration {}'.format(len(positives), b, self._iter_count)
                visualise_tree_traversal(ndx, None, positives, negatives,
                                         '2_processed_leaves_{}_{}.png'.format(self._iter_count, b), limits=limits,
                                         vis=debug)

            clusters.extend(positives)

            if len(positives) > 0:
                print 'Iteration {}. Found {} clusters in beam {}'.format(self._iter_count, len(positives), b)

        print 'Iteration {}. Found {}.Process leaves across 32 beams finished in {} seconds'.format(self._iter_count,
                                                                                                    len(clusters),
                                                                                                    time.time() - t1)
        # todo Cluster the remaining data points by using the mean and slope. Investigate whether to use the slope
        # as another parameter in the clustering technique or by clustering if the joined cluster is not
        # very different from that of the individual data points.

        if not clusters:
            return []
        # Cluster the leaves based on their vicinity to each other
        eps = estimate_leave_eps(clusters)

        log.info('Found {} clusters from {} leaves. EPS is {:0.3f}'.format(len(clusters), len(leaves), eps))

        cluster_data = cluster_leaves(clusters, distance_thold=eps)
        cluster_labels = cluster_data[:, 6]

        visualise_clusters(cluster_data, cluster_labels, np.unique(cluster_labels), None, clusters,
                           filename='3_clusters_{}.png'.format(self._iter_count),
                           limits=limits,
                           debug=debug)

        # Validate the clusters
        valid_tracks = validate_clusters(self.pool, cluster_data, unique_labels=np.unique(cluster_labels))

        visualise_tracks(valid_tracks, None, '5_tracks_{}.png'.format(self._iter_count), limits=limits, debug=debug)

        # Fill needs to be changed to detector.py since which beam will you use to fill?
        # valid_tracks = [fill2(input_data, t, default=np.mean(t[:, 2])) for t in valid_tracks]

        # visualise_post_processed_tracks(tracks, None,
        #                                 '6_post-processed-tracks_{}.png'.format(self._iter_count), limits=None,
        #                                 debug=debug)

        # Diff. Only select data points that are -5 dB from the maximum and 1 dB.
        return [self.diff(input_data, c, channels, channel_noise, obs_info, iter_counter) for c in valid_tracks]

    def diff(self, input_data, cluster, channels, beam_noise, obs_info, iter_counter):
        chl = cluster[:, 0].astype(int)
        sample = cluster[:, 1].astype(int)
        clusters = []
        m = []
        power = np.power(np.abs(input_data[:, chl, sample]), 2.0)
        for beam_id in range(0, 32):
            beam_power = power[beam_id]
            noise_estimate = np.mean(beam_noise[beam_id])

            signal_power = beam_power - noise_estimate

            # Remove data points whose power is below the noise estimate
            s_mask = signal_power > 0
            beam_cluster = cluster[s_mask]

            if len(beam_cluster) < 5:
                continue

            # Select data points whose power is less than 5 db from the peak snr
            beam_snr = 10 * np.log10(signal_power[s_mask] / noise_estimate)

            # Select data points that are less than 5 dB from the peak SNR of the detections
            thres_db = np.max(beam_snr) - 5

            # Minimum SNR is 1 dB
            if thres_db < 1:
                thres_db = 1

            beam_mask = beam_snr >= thres_db

            c = beam_cluster[beam_mask]
            c[:, 2] = beam_snr[beam_mask]

            if len(c) < 5:
                # Add the cluster if it is large enough
                continue

            clusters.append(_create_cluster(c, channels, obs_info, beam_id, iter_counter))
        if len(clusters) > 0:
            m = pd.concat(clusters)

            print 'Cluster was {} long and this became {} after splitting into the beams'.format(len(cluster), len(m))
        return m

    # @timeit
    def _get_clusters_naive(self, input_data, channels, channel_noise, obs_info, iter_counter, limits, debug):
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

        # print 'td', obs_info['sampling_time'], obs_info['channel_bandwidth']

        clusters = []
        for beam_id in range(0, 32):
            # beam_id = 15
            clusters.extend(detect(input_data, channels, t0, td, iter_counter, channel_noise, beam_id))
        log.debug('Found {} new candidates in {} beams'.format(len(clusters), 32))

        return clusters

    def should_debug(self, obs_info):
        expected_transit_time = datetime.datetime.strptime('05 MAR 2019 11:23:28.73', '%d %b %Y %H:%M:%S.%f')
        start_window = expected_transit_time - datetime.timedelta(seconds=30)
        end_window = expected_transit_time + datetime.timedelta(seconds=10)

        print 'Current time is', obs_info['timestamp']
        print 'TLE is between {} and {}'.format(start_window, end_window)
        print 'Detector Within window', end_window >= obs_info['timestamp'] >= start_window, self._iter_count

        if end_window >= obs_info['timestamp'] >= start_window:
            return True

        return False

    def process(self, obs_info, input_data, output_data):
        """
        Sieve through the pre-processed channelised data for space debris candidates

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        obs_info['iter_count'] = self._iter_count
        obs_info['transitted_tracks'] = []
        channels, doppler_mask = self._apply_doppler_mask(obs_info)

        obs_info['doppler_mask'] = doppler_mask

        debug = self.should_debug(obs_info)

        limits = (0, 160, 4000, 4600)

        # with doppler mask
        limits = (0, 160, 1500, 2000)

        # Skip the first few blobs (to allow for an accurate noise estimation to be determined)
        if self._iter_count < 2:
            return obs_info

        self.filter_tx.apply(input_data, obs_info)

        input_data = input_data[:, doppler_mask, :]

        # mean of the doppler mask
        obs_info['mean_noise'] = np.mean(obs_info['channel_noise'][:, doppler_mask])
        obs_info['doppler_mask'] = doppler_mask

        channel_noise = obs_info['channel_noise'][:, obs_info['doppler_mask']]

        # [Feature Extraction] Process the input data and identify the detection clusters
        input_data_org = input_data.copy()

        t = time.time()
        clusters_dbscan = self._get_clusters_naive(input_data_org, channels, channel_noise, obs_info, self._iter_count,
                                                   limits, debug)
        clusters = clusters_dbscan
        print self._iter_count, 'dbscan finished in ', time.time() - t
        t = time.time()
        clusters_msds = self._get_clusters_msds(input_data, channels, channel_noise, obs_info, self._iter_count, limits,
                                                debug)
        print self._iter_count, 'msds finished in ', time.time() - t

        clusters = clusters_msds

        print 'input_data shape', input_data.shape
        # debug = self._iter_count in [4, 5, 6]

        compare_algorithms(input_data_org[14, :, :], msds_clusters=clusters_msds, db_scan_clusters=clusters_dbscan,
                           limits=limits, iteration=self._iter_count, debug=debug)

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

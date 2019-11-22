import datetime
from multiprocessing import Pool

import pandas as pd

from msds.msds import *
from msds.util import _create_cluster
from pybirales import settings
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.pipeline.modules.detection.dbscan_detection import detect
from pybirales.pipeline.modules.detection.filter import RemoveTransmitterChannelFilter
from pybirales.pipeline.modules.detection.util import *


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

        self.name = "MSDS Detector"

        # A list of space debris tracks currently in memory
        self._candidates = []

        self._candidates_msds = []

        # Number of RSO tracks detected in this observation
        self._n_rso = 0
        self._n_rso_msds = 0

        self.filter_tx = RemoveTransmitterChannelFilter()

        self.detections = {}

    def _tear_down(self):
        """

        :return:
        """
        # if settings.detection.multi_proc:
        self.pool.close()

    def _msds_image_segmentation(self, input_data, obs_info):
        # input_data = np.power(np.abs(input_data), 2.0)

        obs_info['beam_noise'] = np.mean(np.mean(input_data, axis=1), axis=1)

        input_data_max = np.max(input_data, axis=(1, 2), keepdims=True)
        input_data_min = np.min(input_data, axis=(1, 2), keepdims=True)

        input_data_norm = (input_data - input_data_min) / (input_data_max - input_data_min)

        threshold = 3 * np.std(input_data_norm, axis=2) + np.mean(input_data_norm, axis=2)

        t2 = np.expand_dims(threshold, axis=2)
        input_data_norm[input_data_norm <= t2] = 0

        input_data = input_data_norm

        return input_data, obs_info

    def _get_clusters_msds(self, input_data, limits, debug):
        """
        Multi-pixel streak detection strategy

        Algorithm combines the beam data across the multi-pixel in order to increase SNR whilst
        reducing the computational load. Data is transformed such that data points belonging
        to the same streak, cluster around a common point.  Then, a noise-aware clustering algorithm,
        such as DBSCAN, can be applied on the data points to identify the candidate tracks.

        :param input_data:
        :param limits:
        :param debug:
        :return:
        """

        clusters = []

        for b in range(0, 32):
            beam_seg_input_data = input_data[b, ...]

            ndx = pre_process_data(beam_seg_input_data, noise_estimate=None)

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

        if not clusters:
            return []

        # Cluster the leaves based on their vicinity to each other
        eps = estimate_leave_eps(clusters)

        log.info('Found {} clusters in iteration {}. EPS is {:0.3f}'.format(len(clusters), self._iter_count, eps))

        cluster_data = cluster_leaves(clusters, distance_thold=eps)
        cluster_labels = cluster_data[:, 6]

        visualise_clusters(cluster_data, cluster_labels, np.unique(cluster_labels), None, clusters,
                           filename='3_clusters_{}.png'.format(self._iter_count),
                           limits=limits,
                           debug=debug)

        # Validate the clusters
        valid_tracks = validate_clusters(self.pool, cluster_data, unique_labels=np.unique(cluster_labels))

        visualise_tracks(valid_tracks, None, '5_tracks_{}.png'.format(self._iter_count), limits=limits, debug=debug)

        return valid_tracks

    def diff(self, input_data, cluster, channels, obs_info, iter_counter, nbeams):
        chl = cluster[:, 0].astype(int)
        sample = cluster[:, 1].astype(int)
        clusters = []
        m = []
        # power = np.power(np.abs(input_data[:, chl, sample]), 2.0)
        power = input_data[:, chl, sample]

        for beam_id in range(0, nbeams):
            beam_power = power[beam_id]
            noise_estimate = np.mean(obs_info['channel_noise'][beam_id])

            # Remove noise power from signal
            signal_power = beam_power - noise_estimate

            # Remove data points whose power is below the noise estimate
            s_mask = signal_power > 0
            beam_cluster = cluster[s_mask]

            if len(beam_cluster) < 5:
                continue

            # Select data points whose power is less than 5 db from the peak snr
            beam_snr = 10 * np.log10(signal_power[s_mask] / noise_estimate)

            # Select data points that are less than 5 dB from the peak SNR of the detections
            # thres_db = np.max(beam_snr) - 15

            # Minimum SNR is 1 dB
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

            log.debug(
                'Cluster was {} long and this became {} after splitting into the beams'.format(len(cluster), len(m)))
        return m

    # @timeit
    def _get_clusters_naive(self, input_data, channels, obs_info, iter_counter, limits, debug):
        """
        Naive algorithm.

        Algorithm uses the DBSCAN algorithm to identify clusters. It iterates over all the beams and considers
        all the pixels that have a non-zero


        :param input_data:
        :param channels:
        :param obs_info:
        :param iter_counter:
        :return:
        """

        # input_data = np.power(np.abs(input_data), 2.0)
        #
        # obs_info['channel_noise'] = np.mean(input_data, axis=2)
        # obs_info['channel_noise_std'] = np.std(input_data, axis=2)

        threshold = 4 * obs_info['channel_noise_std'][:, obs_info['doppler_mask'], ...] + obs_info['channel_noise'][:,
                                                                                          obs_info['doppler_mask'], ...]
        t2 = np.expand_dims(threshold, axis=2)
        input_data[input_data <= t2] = 0

        t0 = np.datetime64(obs_info['timestamp'])
        td = np.timedelta64(int(obs_info['sampling_time'] * 1e9), 'ns')

        clusters = []
        for beam_id in range(0, 32):
            clusters.extend(detect(input_data, channels, t0, td, iter_counter, obs_info['channel_noise'], beam_id))
        log.debug('Found {} new candidates in {} beams'.format(len(clusters), 32))

        return clusters

    def should_debug(self, obs_info, timestamp='05 MAR 2019 11:23:28.73'):
        expected_transit_time = datetime.datetime.strptime(timestamp, '%d %b %Y %H:%M:%S.%f')
        start_window = expected_transit_time - datetime.timedelta(seconds=30)
        end_window = expected_transit_time + datetime.timedelta(seconds=10)

        log.debug('[Visualisation] Current time is {}'.format(obs_info['timestamp']))
        log.debug('[Visualisation] TLE is between {} and {}'.format(start_window, end_window))
        log.debug(
            '[Visualisation] Detector Within window {} {}'.format(end_window >= obs_info['timestamp'] >= start_window,
                                                                  self._iter_count))

        if end_window >= obs_info['timestamp'] >= start_window:
            return True

        return False

    def __process_msds(self, input_data, obs_info, channels, limits, debug):

        # [Image segmentation] Filter input data and get the noise_estimate across N beams
        seg_input_data, obs_info = self._msds_image_segmentation(input_data, obs_info)

        # [Feature Extraction] Process the input data and identify the detection clusters
        clusters = self._get_clusters_msds(seg_input_data, limits, debug)

        # Split the track detection into multiple beams
        clusters = [self.diff(input_data, c, channels, obs_info, self._iter_count, nbeams=32) for c in clusters]

        # [Track Association] Create new tracks from clusters or merge clusters into existing tracks
        candidates = aggregate_clusters(self._candidates_msds, clusters, obs_info,
                                        notifications=settings.observation.notifications,
                                        save_candidates=settings.detection.save_candidates)

        # [Track Termination] Check each track and determine if the detection object has transitted outside FoV
        self._candidates_msds, self._n_rso_msds = active_tracks(candidates, self._n_rso_msds, self._iter_count)

        # Output a TDM for the tracks that have transitted outside the telescope's FoV
        obs_info['transitted_tracks_msds'] = [c for c in candidates if c not in self._candidates_msds]

        for i, candidate in enumerate(self._candidates_msds):
            log.info("MSDS RSO %d: %s", self._n_rso_msds + 1 + i, candidate.state_str())

        return obs_info

    def __process_dbscan(self, input_data, obs_info, channels, limits, debug):

        # [Feature Extraction] Process the input data and identify the detection clusters
        clusters = self._get_clusters_naive(input_data, channels, obs_info,
                                            self._iter_count,
                                            limits, debug)

        # [Track Association] Create new tracks from clusters or merge clusters into existing tracks
        candidates = aggregate_clusters(self._candidates, clusters, obs_info,
                                        notifications=settings.observation.notifications,
                                        save_candidates=settings.detection.save_candidates)

        # [Track Termination] Check each track and determine if the detection object has transitted outside FoV
        self._candidates, self._n_rso = active_tracks(candidates, self._n_rso, self._iter_count)

        # Output a TDM for the tracks that have transitted outside the telescope's FoV
        obs_info['transitted_tracks_msds'] = [c for c in candidates if c not in self._candidates]

        for i, candidate in enumerate(self._candidates):
            log.info("DBSCAN RSO %d: %s", self._n_rso + 1 + i, candidate.state_str())

        return obs_info

    def process(self, obs_info, input_data, output_data):
        """
        Sieve through the pre-processed channelised data for space debris candidates

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """
        # Skip the first few blobs (to allow for an accurate noise estimation to be determined)
        if self._iter_count < 2:
            return obs_info

        self.filter_tx.apply(input_data, obs_info)

        obs_info['iter_count'] = self._iter_count
        obs_info['transitted_tracks'] = []
        self.channels, self._doppler_mask = apply_doppler_mask(self._doppler_mask, self.channels,
                                                               settings.detection.doppler_range,
                                                               obs_info)

        obs_info['doppler_mask'] = self._doppler_mask

        limits = (0, 160, 1500, 2000)  # with doppler mask
        debug = self.should_debug(obs_info, timestamp='05 MAR 2019 11:23:28.73')

        obs_info = self.__process_dbscan(input_data[:, self._doppler_mask, :].copy(), obs_info, self.channels, limits,
                                         debug)
        obs_info = self.__process_msds(input_data[:, self._doppler_mask, :], obs_info, self.channels, limits, debug)

        compare_algorithms(msds_clusters=self._candidates_msds, db_scan_clusters=self._candidates,
                           limits=limits, iteration=self._iter_count, debug=debug)

        return obs_info

    def generate_output_blob(self):
        return ChannelisedBlob(self._config, self._input.shape, datatype=np.float)

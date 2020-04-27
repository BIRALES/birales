import logging
import pickle
import time
from multiprocessing import Pool
from sys import stdout

from skimage.filters import threshold_triangle

from msds.msds import *
from msds.util import _create_cluster
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.pipeline.modules.detection.dbscan_detection import detect
from pybirales.pipeline.modules.detection.filter import RemoveTransmitterChannelFilter
from pybirales.pipeline.modules.detection.util import *
from pybirales.pipeline.modules.persisters.fits_persister import TLE_Target


def triangle_filter(input_data, obs_info):
    for b in range(0, input_data.shape[0]):
        beam_data = input_data[b, ...]
        local_thresh = threshold_triangle(beam_data, nbins=2048)
        local_filter_mask = beam_data < local_thresh

        beam_data[local_filter_mask] = -100

    return input_data


def sigma_clip(input_data, obs_info):
    threshold = 3 * obs_info['channel_noise_std'][:, obs_info['doppler_mask'], ...] + obs_info['channel_noise'][:,
                                                                                      obs_info['doppler_mask'], ...]
    t2 = np.expand_dims(threshold, axis=2)
    input_data[input_data <= t2] = 0

    return input_data


def serial_msds(data):
    t1 = time.time()
    beam_data, b, iter, noise_est = data
    # beam_seg_input_data, noise_est = sigma_clip(beam_data)
    # beam_seg_input_data, noise_est = triangle_filter(beam_data)

    debug = b == 22 and iter == 9  # 41182
    limits = (80, 160, 1750, 1850)  # 41182
    # limits = (80, 115, 3600, 3750)
    limits = None
    debug = False

    true_tracks = None

    ext = '__{}_{}.png'.format(iter, b)

    ndx = pre_process_data(beam_data, noise_estimate=noise_est)

    # Create tree on the merged input data
    k_tree = build_tree(ndx, leave_size=40, n_axis=2)

    visualise_filtered_data(ndx, true_tracks, '1_filtered_data' + ext, limits=limits, debug=debug)

    # Traverse the tree and identify valid linear streaks
    leaves = traverse(k_tree.tree, ndx, bbox=(0, beam_data.shape[1], 0, beam_data.shape[0]), min_length=2.)

    positives = process_leaves(leaves)

    visualise_tree_traversal(ndx, true_tracks, positives, leaves, '2_processed_leaves' + ext, limits=limits,
                             vis=debug)

    if not positives:
        return []
    # Cluster the leaves based on their vicinity to each other
    eps = estimate_leave_eps(positives)

    cluster_data = h_cluster_leaves(positives, distance_thold=eps)

    visualise_clusters(cluster_data, true_tracks, positives,
                       filename='3_clusters' + ext,
                       limits=limits,
                       debug=debug)

    # print 'Beam {}. h_cluster_leaves Finished in {:0.3f}'.format(b, time.time() - t3)
    valid_tracks = validate_clusters(cluster_data, beam_id=b, debug=debug)

    visualise_tracks(valid_tracks, None, '4_tracks' + ext, limits=limits, debug=debug)

    # print 'Beam {}. Noise: {:0.3f}. Finished in {:0.3f}'.format(b, noise_est, time.time() - t1)

    log.info(
        '[Iter {} Detections] Beam {}: {} leaves, {} clusters, {} tracks in {:0.3f}'.format(iter, b, len(positives),
                                                                                            len(cluster_data),
                                                                                            len(valid_tracks),
                                                                                            time.time() - t1))
    return valid_tracks


def _get_clusters_naive(input_data, channels, obs_info, iter_counter):
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

    # threshold = 4 * obs_info['channel_noise_std'][:, obs_info['doppler_mask'], ...] + obs_info['channel_noise'][:,
    #                                                                                   obs_info['doppler_mask'], ...]
    # t2 = np.expand_dims(threshold, axis=2)
    # input_data[input_data <= t2] = 0

    t0 = np.datetime64(obs_info['timestamp'])
    td = np.timedelta64(int(obs_info['sampling_time'] * 1e9), 'ns')

    clusters = []
    for beam_id in range(0, 32):
        clusters.extend(detect(input_data, channels, t0, td, iter_counter, obs_info['channel_noise'], beam_id))
    log.debug('Found {} new candidates in {} beams'.format(len(clusters), 32))

    return clusters


class Detector(ProcessingModule):
    _valid_input_blobs = [ChannelisedBlob]

    def __init__(self, config, input_blob=None):
        # Ensure that the input blob is of the expected format
        self._validate_data_blob(input_blob, valid_blobs=[ChannelisedBlob])

        if settings.detection.multi_proc:
            self.pool = Pool(8)

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

    @timeit
    def _msds_image_segmentation(self, input_data, obs_info):
        obs_info['beam_noise'] = np.mean(np.mean(input_data, axis=1), axis=1)

        local_thresh = threshold_triangle(input_data, nbins=2048)
        local_filter_mask = input_data < local_thresh

        input_data[local_filter_mask] = -100

        return input_data, obs_info

    def diff(self, input_data, clusters, channels, obs_info, iter_counter, nbeams):
        candidates = []
        for cluster in clusters:
            chl = cluster[:, 0].astype(int)
            sample = cluster[:, 1].astype(int)
            beam_clusters = []

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

                beam_clusters.append(_create_cluster(c, channels, obs_info, beam_id, iter_counter))
            if len(beam_clusters) > 0:
                m = pd.concat(beam_clusters)

                log.debug('Cluster was {} long and this became {} after splitting into the beams'.format(len(cluster),
                                                                                                         len(m)))
                candidates.append(m)
        return candidates

    # @timeit

    def should_debug(self, obs_info, timestamp='05 MAR 2019 11:23:28.73'):
        expected_transit_time = datetime.datetime.strptime(timestamp, '%d %b %Y %H:%M:%S.%f')
        start_window = expected_transit_time - datetime.timedelta(seconds=30)
        end_window = expected_transit_time + datetime.timedelta(seconds=10)

        log.debug('[Visualisation] [Iter {}] Current time is {}'.format(self._iter_count, obs_info['timestamp']))
        log.debug('[Visualisation] TLE is between {} and {}'.format(start_window, end_window))
        log.debug(
            '[Visualisation] Detector Within window {} {}'.format(end_window >= obs_info['timestamp'] >= start_window,
                                                                  self._iter_count))

        if end_window >= obs_info['timestamp'] >= start_window:
            return True

        return False

    @timeit
    def __process_msds(self, input_data, obs_info, channels, limits, debug):

        # [Image segmentation] Filter input data and get the noise_estimate across N beams
        # seg_input_data, obs_info = self._msds_image_segmentation(input_data, obs_info)
        # root = 'data'
        # if self._iter_count == 2:

        # np.save('data/input_data_{}.npy'.format(self._iter_count), input_data)
        # pickle.dump(obs_info, open('obs_info_{}.pkl'.format(self._iter_count), "wb"))
        # np.save('channels_{}.npy'.format(self._iter_count), channels)

        # [Feature Extraction] Process the input data and identify the detection clusters
        clusters = _get_clusters_msds(self.pool, input_data, obs_info, self._iter_count, limits, debug)

        # [Track Association] Create new tracks from clusters or merge clusters into existing tracks
        candidates = aggregate_clusters(self._candidates_msds, clusters, obs_info,
                                        notifications=settings.observation.notifications,
                                        save_candidates=settings.detection.save_candidates)

        # [Track Termination] Check each track and determine if the detection object has transitted outside FoV
        self._candidates_msds, self._n_rso_msds = active_tracks(obs_info, candidates, self._n_rso_msds,
                                                                self._iter_count)

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
        self._candidates, self._n_rso = active_tracks(obs_info, candidates, self._n_rso, self._iter_count)

        # Output a TDM for the tracks that have transitted outside the telescope's FoV
        obs_info['transitted_tracks'] = [c for c in candidates if c not in self._candidates]

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

        obs_info['iter_count'] = self._iter_count
        obs_info['transitted_tracks'] = []
        obs_info['transitted_tracks_msds'] = []

        # Skip the first few blobs (to allow for an accurate noise estimation to be determined)
        if self._iter_count < 2:
            return obs_info

        self.filter_tx.apply(input_data, obs_info)

        self.channels, self._doppler_mask = apply_doppler_mask(self._doppler_mask, self.channels,
                                                               settings.detection.doppler_range,
                                                               obs_info)

        obs_info['doppler_mask'] = self._doppler_mask

        if True:
            obs_name = settings.observation.name
            obs_name = 'norad_41128'
            out_dir = os.path.join(os.environ['HOME'], '.birales/debug/detection', obs_name)
            self.dump_detection_data(input_data, obs_info, out_dir)

            return obs_info
        else:

            obs_info = self.__process_msds(input_data[:, self._doppler_mask, :], obs_info, self.channels, limits, debug)
            # compare_algorithms(msds_clusters=self._candidates_msds, db_scan_clusters=self._candidates,
            #                    limits=limits, iteration=self._iter_count, debug=debug)

        return obs_info

    def generate_output_blob(self):
        return ChannelisedBlob(self._config, self._input.shape, datatype=np.float)

    def dump_detection_data(self, input_data, obs_info, out_dir):

        if self._iter_count < 3:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            pickle.dump(self.channels, open('{}/channels.pkl'.format(out_dir, self._iter_count), "wb"))

        pickle.dump(obs_info, open('{}/obs_info_{}.pkl'.format(out_dir, self._iter_count), "wb"))

        np.save('{}/{}_{}.pkl'.format(out_dir, 'input_data', self._iter_count), input_data[:, self._doppler_mask, :])

        return obs_info


def msds_standalone(pool, _iter_count, input_data, obs_info, channels):
    t0 = np.datetime64(obs_info['timestamp'])
    td = np.timedelta64(int(obs_info['sampling_time'] * 1e9), 'ns')
    print "\nIteration {}. From {} to {}".format(_iter_count, str(t0), str(t0 + td * 160))

    # [Feature Extraction] Process the input data and identify the detection clusters
    beam_clusters = []
    # for b in [22, 23]:
    #     beam_clusters += serial_msds((input_data[b, ...], b, _iter_count, np.mean(obs_info['channel_noise'][b])))

    for c in pool.map(serial_msds, [(input_data[b, ...], b, _iter_count, np.mean(obs_info['channel_noise'][b])) for b in
                                    range(0, 32)]):
        beam_clusters += c

    new_tracks = [
        create_candidate(c, channels, _iter_count, 160, t0, td, obs_info['channel_noise'], int(c[:, 4][0])) for c in
        beam_clusters]

    return new_tracks


def dbscan_standalone(pool, _iter_count, input_data, obs_info, channels):
    beam_clusters = []
    # for c in pool.map(_get_clusters_naive, [(input_data[b, ...], b, _iter_count) for b in range(0, 32)]):
    #     beam_clusters += c

    # [Feature Extraction] Process the input data and identify the detection clusters
    beam_clusters = _get_clusters_naive(input_data, channels, obs_info, _iter_count)
    return beam_clusters


if __name__ == '__main__':
    log = logging.getLogger('')
    log.setLevel(logging.DEBUG)
    str_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(stdout)
    ch.setFormatter(str_format)
    log.addHandler(ch)

    pool = Pool(12)
    root = '/home/denis/.birales/debug/detection/'
    detection = msds_standalone
    filtering = triangle_filter

    # detection = dbscan_standalone
    # filtering = sigma_clip

    targets = [
        TLE_Target(name='norad_41128', transit_time='05 MAR 2019 11:23:28.73', doppler=-3226.36329),  # iteration 9
        TLE_Target(name='norad_1328', transit_time='05 MAR 2019 10:37:53.01', doppler=-1462.13774)
    ]

    # Tracks, that are valid and have transitted.
    terminated_tracks = []

    # Track which are currently valid and have not transitted/terminated yet
    pending_tracks = []

    # Tracks that were valid but validation failed later on
    cancelled_tracks = []

    target = targets[1]
    in_dir = os.path.join(root, target.name)
    channels = pickle.load(open(os.path.join(in_dir, 'channels.pkl'), 'rb'))
    for _iter_count in range(4, 9):
        t0 = time.time()
        input_data = np.load(os.path.join(in_dir, 'input_data_{}.pkl.npy'.format(_iter_count)))
        obs_info = pickle.load(open(os.path.join(in_dir, 'obs_info_{}.pkl'.format(_iter_count)), 'rb'))
        tr = time.time() - t0
        t1 = time.time()

        filtered_data = filtering(input_data, obs_info)

        # plot_TLE(obs_info, input_data, tle_target)
        new_tracks = detection(pool, _iter_count, filtered_data, obs_info, channels)

        log.info('[Iter {}]. Found {} beam_candidates'.format(_iter_count, len(new_tracks)))

        # [Track Association] Create new tracks from clusters or merge clusters into existing
        tracks = data_association(pending_tracks, new_tracks, obs_info, notifications=False, save_candidates=False)

        # [Track Termination] Check each track and determine if the detection object has transitted outside FoV
        tracks = active_tracks(obs_info, tracks, _iter_count)

        # reset pending tracks
        pending_tracks = []
        for t in tracks:
            if t.cancelled:
                cancelled_tracks.append(t)
            elif t.terminated:
                terminated_tracks.append(t)
            else:
                pending_tracks.append(t)

        total = len(pending_tracks) + len(terminated_tracks) + len(cancelled_tracks)

        log.info('[Iteration {:d}]. Pending {:d}, Terminated: {:d}, Cancelled: {:d}. Total: {:d}' \
                 .format(_iter_count, len(pending_tracks), len(terminated_tracks), len(cancelled_tracks), total))

        print "Iteration {}: Processing finished in {:0.3f}. Reading took {:0.3f}".format(_iter_count, time.time() - t1,
                                                                                          tr)

    pool.close()

    algo_name = detection.__name__.title()

    # Show pending and terminated tracks
    tracks = pending_tracks + terminated_tracks
    for j, candidate in enumerate(tracks):
        i = j + 1
        log.info("%s RSO %d: %s" % (algo_name, i, candidate.state_str()))
        plot_RSO_track(candidate, "RSO_{}".format(i))

    plot_all_RSOs_track(tracks)
    # plt.interactive(False)
    # plt.show(block=False)

import itertools
import logging as log
import os
import time
from multiprocessing import Pool
import numpy as np, struct
from astropy.time import Time, TimeDelta
from sklearn import linear_model
from sklearn.cluster import DBSCAN

from pybirales.base import settings
from pybirales.modules.detection.beam import Beam
from pybirales.modules.detection.detection_clusters import DetectionCluster
from pybirales.modules.detection.strategies.strategies import SpaceDebrisDetectionStrategy
from pybirales.plotters.spectrogram_plotter import plotter
from functools import partial
from pybirales.modules.detection.queue import BeamCandidatesQueue


_eps = 5
_min_samples = 5
_algorithm = 'kd_tree'
_r2_threshold = 0.90
_merge_threshold = 0.10
_n_proc = 32
_linear_model = linear_model.RANSACRegressor(linear_model.LinearRegression())
db_scan = DBSCAN(eps=_eps, min_samples=_min_samples, algorithm=_algorithm, n_jobs=-1)
_ref_time = None
_time_delta = None



class MultiProcessingDBScanDetectionStrategy(SpaceDebrisDetectionStrategy):
    name = 'Spirit'

    def __init__(self):
        SpaceDebrisDetectionStrategy.__init__(self)
        self.pool = Pool()

    def detect(self, obs_info, input_data):
        global _time_delta, _ref_time
        ref_time = Time(obs_info['timestamp'])
        time_delta = TimeDelta(obs_info['sampling_time'], format='sec')

        beams = [Beam(beam_id=n_beam, obs_info=obs_info, beam_data=input_data)
                 for n_beam in range(settings.detection.beam_range[0], settings.detection.beam_range[1])]

        # Process the beam data to detect the beam candidates
        beam_candidates = []
        try:
            if settings.detection.multi_proc:
                func = partial(m_detect, ref_time, time_delta)
                beam_candidates = self.pool.map(func, beams)

            else:
                for beam in beams:
                    beam_candidates.append(m_detect(beam))
        except Exception:
            log.exception('An exception has occurred')

        # Flatten list of beam candidates returned by the N threads
        return [candidate for sub_list in beam_candidates for candidate in sub_list if candidate]


def _create_clusters(beam):
    """
    Use the DBScan algorithm to create a set of clusters from the given beam data
    :param beam: The beam object from which the clusters will be generated
    :return:
    """

    # Select the data points that are non-zero and transform them in a time (x), channel (y) nd-array
    data = np.column_stack(np.where(beam.snr > 0))

    if not np.any(data):
        return []

    # Perform clustering on the data and returns cluster labels the points are associated with
    try:
        t = time.time()
        cluster_labels = db_scan.fit_predict(data)
        log.debug('Scipy\'s DBScan took %0.3f s', time.time() - t)
    except ValueError:
        log.exception('DBSCAN failed in beam %s', beam.id)
        return []

    # plotter.plot_detections(beam, 'detection/db_scan/' + str(beam.id) + '_' + str(beam.t_0), beam.id == 3,
    #                         cluster_labels=cluster_labels)

    # Select only those labels which were not classified as noise (-1)
    filtered_cluster_labels = cluster_labels[cluster_labels > -1]

    log.debug('Process %s: beam %s, %s out of %s data points was considered to be noise',
              os.getpid(),
              beam.id,
              len(cluster_labels) - len(filtered_cluster_labels),
              len(cluster_labels))

    log.debug('%s unique clusters were detected', len(np.unique(filtered_cluster_labels)))

    # Group the data points in clusters
    clusters = []
    tt = time.time()
    for label in np.unique(filtered_cluster_labels):
        data_indices = data[np.where(cluster_labels == label)]

        log.debug('beam %s: cluster %s contains %s data points', beam.id, label, len(data_indices[:, 1]))

        channel_indices = data_indices[:, 1]
        time_indices = data_indices[:, 0]

        # If cluster is 'small' do not consider it
        if len(channel_indices) < _min_samples:
            log.debug('Ignoring small cluster with %s data points', len(channel_indices))
            continue

        # Create a Detection Cluster from the cluster data
        x = time.time()
        cluster = DetectionCluster(model=_linear_model,
                                   beam=beam,
                                   indices=[time_indices, channel_indices],
                                   time_data=[_ref_time + t * _time_delta for t in beam.time[time_indices]],
                                   channels=beam.channels[channel_indices],
                                   snr=beam.snr[(time_indices, channel_indices)])
        tt = time.time() - x
        log.debug('P took %0.3f s', tt)

        if tt > 2:
            print(beam.channels[channel_indices])
            print(beam.time[time_indices])

        # Add only those clusters that are linear
        if cluster.is_linear(threshold=0.9):
            log.debug('Cluster with m:%3.2f, c:%3.2f, n:%s and r:%0.2f is considered to be linear.', cluster.m,
                      cluster.c, len(channel_indices), cluster.score)
            clusters.append(cluster)
    log.debug('%s candidates (%s valid) created in %0.3f s',
              len(np.unique(filtered_cluster_labels)),
              len(clusters),
              time.time() - tt)

    # plotter.plot_detections(beam, 'detection/db_scan/' + str(beam.id) + '_' + str(beam.t_0), beam.id == 3,
    #                         clusters=clusters)

    log.debug('DBSCAN detected %s clusters in beam %s, of which %s are linear',
              len(np.unique(filtered_cluster_labels)),
              beam.id, len(clusters))

    return clusters


def _merge_clusters(clusters):
    """
    Merge clusters based on how similar the gradient and y-intercept are

    :param clusters: The clusters that will be evaluated for merge
    :type clusters: list of DetectionCandidates
    :return:
    """

    if not clusters:
        log.debug("No clusters to merge")
        return clusters

    done = False
    while not done:
        done = True
        for cluster_1, cluster_2 in itertools.combinations(clusters, 2):
            if cluster_1.is_similar_to(cluster_2, _merge_threshold):
                # Create a new merged cluster if clusters are similar
                merged_cluster = cluster_1.merge(cluster_2)

                # Append the new cluster to the list
                clusters.append(merged_cluster)

                # Delete old clusters
                clusters = [cluster for cluster in clusters if cluster not in [cluster_1, cluster_2]]

                # Re-compare the cluster with the rest
                done = False

                # Break loop to re-compare
                break

    return clusters


def m_detect(ref_time, time_delta, beam):
    """
    The core detection algorithm to be applied to the incoming data

    To be run in parallel using the multi-processing queue

    :param beam:
    :return:
    """

    global _time_delta, _ref_time
    _ref_time = ref_time
    _time_delta = time_delta

    # Apply the pre-processing filters to the beam data
    candidates = []

    try:
        # plotter = SpectrogramPlotter()
        t = time.time()
        beam.apply_filters()
        log.debug('Filters took %0.3f s', time.time() - t)
        # plotter.plot(beam, 'detection/filtered_beam/' + str(beam.id) + '_' + str(beam.t_0), beam.id == 3)

        # Run detection algorithm on the beam data to extract possible candidates
        t = time.time()
        clusters = _create_clusters(beam)
        log.debug('Candidates created in %0.3f s', time.time() - t)

        # Merge clusters which are identified as being similar
        t2 = time.time()
        candidates = _merge_clusters(clusters)
        log.debug('Merging of clusters took %0.3f s', time.time() - t2)

        t3 = time.time()
        _debris_queue = BeamCandidatesQueue(_n_proc)
        for candidate in candidates:
            _debris_queue.enqueue(candidate)
        log.debug('Merging of clusters took %0.3f s', time.time() - t3)

    except Exception:
        log.exception('Something went wrong with process')

    return candidates


if __name__ == "__main__":
    pass

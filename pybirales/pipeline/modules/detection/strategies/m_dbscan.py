import logging as log
import os
import time

import numpy as np
from pybirales import settings
from pybirales.pipeline.modules.detection.detection_clusters import DetectionCluster
from pybirales.repository.repository import BeamCandidateRepository
from sklearn import linear_model
from sklearn.cluster import DBSCAN

from pybirales.pipeline.plotters.spectrogram_plotter import plotter1

_eps = 5
_min_samples = 5
_algorithm = 'kd_tree'
_r2_threshold = 0.90
_merge_threshold = 0.20
_n_proc = 32
_linear_model = linear_model.RANSACRegressor(linear_model.LinearRegression())
db_scan = DBSCAN(eps=_eps, min_samples=_min_samples, algorithm=_algorithm, n_jobs=-1)
_ref_time = None
_time_delta = None


def _create_clusters(beam):
    """
    Use the DBScan algorithm to create a set of clusters from the given beam data
    :param beam: The beam object from which the clusters will be generated
    :return:
    """

    global r

    # Select the data points that are non-zero and transform them in a time (x), channel (y) nd-array
    ndx = np.where(beam.snr > 0)

    data = np.column_stack(ndx)
    # c_data = np.hstack((data, beam.snr[ndx].reshape(ndx[0].shape[0], 1)))

    if not np.any(data):
        return []

    # Perform clustering on the data and returns cluster labels the points are associated with
    try:
        t = time.time()
        cluster_labels = db_scan.fit_predict(data)
        log.debug('Beam %s: Scipy\'s DBScan took %0.3f s', beam.id, time.time() - t)
    except ValueError:
        log.exception('DBSCAN failed in beam %s', beam.id)
        return []

    # if settings.detection.debug_candidates:
    #     plotter.plot_detections(beam, 'detection/db_scan/' + str(beam.id) + '_' + str(_ref_time), beam.id == 15,
    #                             cluster_labels=cluster_labels, cluster_data=data)

    # Select only those labels which were not classified as noise (-1)
    filtered_cluster_labels = cluster_labels[cluster_labels > -1]

    log.debug('Process %s: beam %s, %s out of %s data points was considered to be noise',
              os.getpid(),
              beam.id,
              len(cluster_labels) - len(filtered_cluster_labels),
              len(cluster_labels))

    log.debug('Beam %s: %s unique clusters were detected', beam.id, len(np.unique(filtered_cluster_labels)))

    # Group the data points in clusters
    clusters = []
    tt = time.time()

    def check_doppler_range(channels):
        dp = (channels - beam.tx) * 1e6
        return (dp > settings.detection.doppler_range[0]).all() and (dp < settings.detection.doppler_range[1]).all()

    for label in np.unique(filtered_cluster_labels):
        data_indices = data[np.where(cluster_labels == label)]

        log.debug('Beam %s: cluster %s contains %s data points', beam.id, label, len(data_indices[:, 1]))

        channel_indices = data_indices[:, 1]
        time_indices = data_indices[:, 0]

        # If cluster is 'small' do not consider it
        if len(channel_indices) < _min_samples:
            log.debug('Ignoring small cluster with %s data points', len(channel_indices))
            continue

        if not check_doppler_range(beam.channels[channel_indices]):
            continue

        # Create a Detection Cluster from the cluster data
        tr = time.time()
        a = [_ref_time + t * _time_delta for t in beam.time[time_indices]]

        trd = time.time() - tr

        x = time.time()
        cluster = DetectionCluster(model=_linear_model,
                                   beam_config=beam.get_config(),
                                   time_data=[_ref_time + t * _time_delta for t in beam.time[time_indices]],
                                   channels=beam.channels[channel_indices],
                                   snr=beam.snr[(time_indices, channel_indices)])
        tt = time.time() - x
        log.debug('T/P took %0.3f s', trd / tt)

        # Add only those clusters that are linear
        if cluster.is_linear(threshold=0.9) and cluster.is_valid():
            log.debug('Cluster with m:%3.2f, c:%3.2f, n:%s and r:%0.2f is considered to be linear.', cluster.m,
                      cluster.c, len(channel_indices), cluster.score)
            clusters.append(cluster)

    log.debug('Beam %s: %s candidates (%s valid) created in %0.3f s',
              beam.id,
              len(np.unique(filtered_cluster_labels)),
              len(clusters),
              time.time() - tt)

    log.debug('DBSCAN detected %s clusters in beam %s, of which %s are linear',
              len(np.unique(filtered_cluster_labels)),
              beam.id, len(clusters))

    return clusters


def m_detect(obs_info, queue, beam):
    """
    The core detection algorithm to be applied to the incoming data

    To be run in parallel using the multi-processing queue

    :param ref_time:
    :param time_delta:
    :param beam:
    :return:
    """
    # ref_time = Time(obs_info['timestamp'])
    # time_delta = TimeDelta(obs_info['sampling_time'], format='sec')

    ref_time = np.datetime64(obs_info['timestamp'])

    time_delta = np.timedelta64(int(obs_info['sampling_time']*1e9), 'ns')

    # print(obs_info['timestamp'])
    #
    # print(datetime.datetime.strptime(obs_info['timestamp'], "%Y-%m-%dT%H:%M:%S.%f"))

    global _time_delta, _ref_time, _debris_queue, r
    _ref_time = ref_time
    _time_delta = time_delta

    queue.set_repository(BeamCandidateRepository())

    # Apply the pre-processing filters to the beam data
    try:
        t1 = time.time()
        # if settings.detection.debug_candidates and beam.id == 11:
        #     plotter1.plot(beam, 'detection/input_beam/' + str(beam.id) + '_' + str(_ref_time), beam.id == 11)
        beam.apply_filters()

        # if settings.detection.debug_candidates:
        #     plotter2.plot(beam, 'detection/filtered_beam/' + str(beam.id) + '_' + str(_ref_time), beam.id == 0)

        candidates = _create_clusters(beam)
        log.debug('Beam %s: Create clusters took %0.3f s', beam.id, time.time() - t1)

        t3 = time.time()
        for candidate in candidates:
            queue.enqueue(candidate)
        log.debug('Beam %s: Enqueuing of %s clusters took %0.3f s', beam.id, len(candidates), time.time() - t3)

    except Exception:
        log.exception('Something went wrong with process')

    return queue.get_candidates(beam.id)

import itertools
import logging as log
import os
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import DBSCAN_multiplex as DB2
import numpy as np
from astropy.time import Time, TimeDelta
from sklearn import linear_model
from sklearn.cluster import DBSCAN

from pybirales.base import settings
from pybirales.modules.detection.beam import Beam
from pybirales.modules.detection.detection_clusters import DetectionCluster
from pybirales.modules.detection.detection_strategies import SpaceDebrisDetectionStrategy
from pybirales.plotters.spectrogram_plotter import plotter


class DetectionStrategy(SpaceDebrisDetectionStrategy):
    name = 'Spirit'
    _eps = 5
    _min_samples = 10
    _algorithm = 'kd_tree'
    _r2_threshold = 0.90
    _merge_threshold = 0.10

    def __init__(self):
        SpaceDebrisDetectionStrategy.__init__(self)
        # Initialise the clustering algorithm
        self.db_scan = DBSCAN(eps=self._eps,
                              min_samples=self._min_samples,
                              algorithm=self._algorithm,
                              # metric="precomputed",
                              n_jobs=-1)

        self._m_pool = Pool()

        self._linear_model = linear_model.RANSACRegressor(linear_model.LinearRegression())

    def _rms(self, data):
        return np.sqrt(np.mean(data ** 2.0))

    def _power(self, data):
        return np.abs(data) ** 2

    def pre_process(self, input_data):

        # SNR calculation
        data = input_data[0, :, :, :].T
        p_v = self._power(data)
        p_n = self._power(self._rms(data))
        data = (p_v - p_n) / p_n
        data[data < 0] = np.nan
        data = 10 * np.log10(data)
        data[np.isnan(data)] = 0.

        # Background noise filter
        mean = np.mean(data)
        std = np.std(data)
        threshold = 2.0 * std + mean
        data[data < threshold] = 0.

        # Transmitter filter
        sum_across_time = data.sum(axis=1)
        peaks = np.where(sum_across_time > 20.0)[0]

        for i, peak in enumerate(peaks):
            peak_snr = data[peak]
            mean = np.mean(peak_snr[peak_snr > 0.0])
            data[peak][peak_snr > 0.0] -= mean
            data[peak][peak_snr < 0.0] = 0.0

        return data

    def process(self, input_data):
        pass

    def detect(self, obs_info, input_data):
        # Pre-process the input data
        input_data = self.pre_process(input_data)

        # Process the input data - create clusters from the input data
        clusters = self._process(obs_info, input_data)

        # Post-process the clusters
        clusters = self.pre_process(clusters)

        return clusters

    def _process(self, obs_info, input_data):
        # Select the data points that are non-zero and transform them in a time (x), channel (y) nd-array
        data = np.column_stack(np.where(input_data > 0))

        try:
            cluster_labels = self.db_scan.fit_predict(data)
        except ValueError:
            log.exception('DBSCAN failed')
            return []

        # Select only those labels which were not classified as noise (-1)
        filtered_cluster_labels = cluster_labels[cluster_labels > -1]

        if len(np.unique(filtered_cluster_labels)) < 1:
            return []

        log.debug('%s unique clusters were detected', len(np.unique(filtered_cluster_labels)))

        return self._create_clusters(obs_info, input_data, cluster_labels)

    def _create_clusters(self, obs_info, input_data, cluster_labels):
        clusters = []
        ref_time = Time(obs_info['timestamp'])
        dt = TimeDelta(obs_info['sampling_time'], format='sec')

        for label in np.unique(cluster_labels):
            data_indices = input_data[np.where(cluster_labels == label)]

            channel_indices = data_indices[:, 1]
            time_indices = data_indices[:, 0]

            # If cluster is 'small' do not consider it
            if len(channel_indices) < self._min_samples:
                log.debug('Ignoring small cluster with %s data points', len(channel_indices))
                continue

            # Create a Detection Cluster from the cluster data
            # cluster = DetectionCluster(model=self._linear_model,
            #                            beam=beam,
            #                            time_data=[ref_time + t * dt for t in beam.time[time_indices]],
            #                            channels=beam.channels[channel_indices],
            #                            snr=beam.snr[(time_indices, channel_indices)])
            #
            # # Add only those clusters that are linear
            # if cluster.is_linear(threshold=0.9):
            #     log.debug('Cluster with m:%3.2f, c:%3.2f, n:%s and r:%0.2f is considered to be linear.', cluster.m,
            #               cluster.c, len(channel_indices), cluster._score)
            #     clusters.append(cluster)

        # log.debug('DBSCAN detected %s clusters in beam %s, of which %s are linear',
        #           len(np.unique(filtered_cluster_labels)),
        #           beam.id, len(clusters))

        return clusters

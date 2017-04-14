import itertools
import logging as log
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import os

from abc import abstractmethod
from pybirales.modules.detection.detection_clusters import DetectionCluster
from sklearn.cluster import DBSCAN
from pybirales.base import settings
import time
from pybirales.plotters.spectrogram_plotter import plotter
from sklearn import linear_model
from astropy.time import Time, TimeDelta
from sklearn.metrics.pairwise import euclidean_distances
import hdbscan
import DBSCAN_multiplex as DB2


class SpaceDebrisDetection(object):
    name = None

    def __init__(self, detection_strategy_name):
        try:
            self.detection_strategy = globals()[detection_strategy_name]()
            self.name = self.detection_strategy.name
        except KeyError:
            log.error('%s is not a valid detection strategy. Exiting.', detection_strategy_name)
            sys.exit()

    def detect(self, beam):
        candidates = self.detection_strategy.detect(beam)
        return candidates


class SpaceDebrisDetectionStrategy(object):
    def __init__(self):
        self.max_detections = settings.detection.max_detections

    @abstractmethod
    def detect(self, beam):
        pass


class SpiritSpaceDebrisDetectionStrategy(SpaceDebrisDetectionStrategy):
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
                              # algorithm=self._algorithm,
                              metric="precomputed",
                              n_jobs=-1)
        self.db_scan = hdbscan.HDBSCAN(min_cluster_size=self._min_samples)

        self._linear_model = linear_model.RANSACRegressor(linear_model.LinearRegression())

    def detect(self, beam):
        t2 = time.time()
        clusters = self._create_clusters(beam)
        log.debug("%s detection clusters detected in beam %s in %0.2f", len(clusters), beam.id, time.time() - t2)

        t3 = time.time()
        merged_clusters = self._merge_clusters(clusters)
        log.debug("%s detection clusters remain after merging in beam %s in %0.2f", len(merged_clusters), beam.id,
                  time.time() - t3)

        return merged_clusters

    def c(self, data):
        t1 = time.time()
        cluster_labels = self.db_scan.fit_predict(data)
        # log.warning('DBSCAN finished in %1.2f s', time.time() - t1)
        return cluster_labels

    def c2(self, data):
        t1 = time.time()
        e, l = DB2.DBSCAN(data, minPts=5, eps=3.0, verbose=True)
        # log.warning('DBSCAN2 finished in %1.2f s', time.time() - t1)
        return l

    def _create_clusters(self, beam):
        """
        Use the DBScan algorithm to create a set of clusters from the given beam data
        :param beam: The beam object from which the clusters will be generated
        :return:
        """
        t4 = time.time()
        # Select the data points that are non-zero and transform them in a time (x), channel (y) nd-array
        data = np.column_stack(np.where(beam.snr > 0))

        if not np.any(data):
            return []

        # X = euclidean_distances(data, data)
        X = data
        # Perform clustering on the data and returns cluster labels the points are associated with
        t5 = time.time()
        try:

            cluster_labels = self.c(X)

        # cluster_labels = self.c2(X)
        except ValueError:
            log.exception('DBSCAN failed in beam %s', beam.id)
            return []
        log.debug('DBSCAN finished in %1.3f s in beam %s', time.time() - t5, beam.id)
        plotter.plot(beam.snr, 'detection/detection_db_scan_' + str(time.time()), beam.id == 0,
                     cluster_labels=cluster_labels)

        # Select only those labels which were not classified as noise (-1)
        filtered_cluster_labels = cluster_labels[cluster_labels > -1]

        log.debug('Process %s: beam %s, %s out of %s data points was considered to be noise',
                  os.getpid(),
                  beam.id,
                  len(cluster_labels) - len(filtered_cluster_labels),
                  len(cluster_labels))

        ref_time = Time(beam.t_0)
        dt = TimeDelta(beam.dt, format='sec')

        log.debug('%s unique clusters were detected', len(np.unique(filtered_cluster_labels)))

        # Group the data points in clusters
        clusters = []
        for label in np.unique(filtered_cluster_labels):
            data_indices = data[np.where(cluster_labels == label)]

            log.debug('beam %s: cluster %s contains %s data points', beam.id, label, len(data_indices[:, 1]))

            channel_indices = data_indices[:, 1]
            time_indices = data_indices[:, 0]

            # If cluster is 'small' do not consider it
            if len(channel_indices) < self._min_samples:
                log.debug('Ignoring small cluster with %s data points', len(channel_indices))
                continue

            # Create a Detection Cluster from the cluster data
            cluster = DetectionCluster(model=self._linear_model,
                                       beam=beam,
                                       time_data=[ref_time + t * dt for t in beam.time[time_indices]],
                                       channels=beam.channels[channel_indices],
                                       snr=beam.snr[(time_indices, channel_indices)])

            # Add only those clusters that are linear
            if cluster.is_linear(threshold=0.9):
                log.debug('Cluster with m:%3.2f, c:%3.2f, n:%s and r:%0.2f is considered to be linear.', cluster.m,
                          cluster.c, len(channel_indices), cluster._score)
                clusters.append(cluster)

        log.debug('DBSCAN detected %s clusters in beam %s, of which %s are linear in %0.3f',
                  len(np.unique(filtered_cluster_labels)),
                  beam.id, len(clusters),
                  time.time() - t4)

        return clusters

    def _merge_clusters(self, clusters):
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
                if cluster_1.is_similar_to(cluster_2, self._merge_threshold):
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

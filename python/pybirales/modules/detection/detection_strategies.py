import itertools
import logging as log
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np

from abc import abstractmethod
from pybirales.modules.detection.detection_clusters import DetectionCluster
from sklearn.cluster import DBSCAN
from pybirales.base import settings
import time
from pybirales.plotters.spectrogram_plotter import plotter
from sklearn import linear_model
from astropy.time import Time, TimeDelta


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
    _eps = 5.0
    _min_samples = 5
    _algorithm = 'kd_tree'
    _r2_threshold = 0.90
    _merge_threshold = 0.10

    def __init__(self):
        SpaceDebrisDetectionStrategy.__init__(self)
        # Initialise the clustering algorithm
        self.db_scan = DBSCAN(eps=self._eps,
                              min_samples=self._min_samples,
                              algorithm=self._algorithm,
                              n_jobs=1)

        self._linear_model = linear_model.RANSACRegressor(linear_model.LinearRegression())

    def detect(self, beam):
        clusters = self._create_clusters(beam)
        log.debug("%s detection clusters detected in beam %s", len(clusters), beam.id)

        merged_clusters = self._merge_clusters(clusters)
        log.debug("%s detection clusters remain after merging in beam %s", len(merged_clusters), beam.id)

        return merged_clusters

    def _create_clusters(self, beam):
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
            cluster_labels = self.db_scan.fit_predict(data)
        except ValueError:
            log.warning('DBSCAN failed')
            return []

        plotter.plot(beam.snr, 'detection/detection_db_scan_' + str(time.time()), beam.id == 6,
                     cluster_labels=cluster_labels)

        # Select only those labels which were not classified as noise (-1)
        filtered_cluster_labels = cluster_labels[cluster_labels > -1]

        ref_time = Time(beam.t_0)
        dt = TimeDelta(beam.dt, format='sec')

        # Group the data points in clusters
        clusters = []
        for label in np.unique(filtered_cluster_labels):
            data_indices = data[np.where(cluster_labels == label)]

            channel_indices = data_indices[:, 1]
            time_indices = data_indices[:, 0]

            # Create a Detection Cluster from the cluster data
            cluster = DetectionCluster(model=self._linear_model,
                                       beam=beam,
                                       time_data=[ref_time + t * dt for t in beam.time[time_indices]],
                                       channels=beam.channels[channel_indices],
                                       snr=beam.snr[(time_indices, channel_indices)])

            # Add only those clusters that are linear
            if cluster.is_linear(threshold=0.9):
                clusters.append(cluster)

        log.debug('DBSCAN detected %s clusters in beam %s, of which %s are linear',
                  len(np.unique(filtered_cluster_labels)),
                  beam.id, len(clusters))

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

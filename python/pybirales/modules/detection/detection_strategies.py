import itertools
import logging as log
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np

from abc import abstractmethod
from pybirales.modules.detection.detection_clusters import DetectionCluster
from pybirales.modules.detection.detection_candidates import BeamSpaceDebrisCandidate
from sklearn.cluster import DBSCAN
from pybirales.base import settings


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

    @staticmethod
    def _create_space_debris_candidates(beam, clusters):
        """
        Create space debris candidates from the clusters data

        :param beam:
        :param clusters:
        :return:
        """
        candidates = []
        beam_candidates_counter = {}

        for cluster in clusters:
            channel_mask = cluster.data[:, 1]
            time_mask = cluster.data[:, 0]

            channels = beam.channels[channel_mask]
            time = beam.time[time_mask]
            snr = beam.snr[time_mask, channel_mask]

            detection_data = np.column_stack((channels, time, snr))
            if beam.id not in beam_candidates_counter:
                beam_candidates_counter[beam.id] = 0
            beam_candidates_counter[beam.id] += 1

            candidate_name = str(beam.id) + '.' + str(beam_candidates_counter[beam.id])
            candidate = BeamSpaceDebrisCandidate(candidate_name, beam, detection_data)
            candidates.append(candidate)
        return candidates


class NaiveDBScanSpaceDebrisDetectionStrategy(SpaceDebrisDetectionStrategy):
    name = 'Naive DBScan'

    def __init__(self):
        SpaceDebrisDetectionStrategy.__init__(self)
        pass

    @staticmethod
    def _merge_clusters_data(cluster1, cluster2):
        return cluster1['data'] + cluster2['data']

    @staticmethod
    def _db_scan_cluster(data):
        """
        Use the DBScan algorithm to create a set of clusters from the given beam data
        :param data:
        :return:
        """
        data = np.transpose(np.nonzero(data > 0.))
        y_pred = DBSCAN(eps=10.0, min_samples=5, algorithm='kd_tree').fit_predict(data)
        clusters = dict.fromkeys(np.unique(y_pred))

        # Classify coordinates into corresponding clusters
        for i, y in enumerate(y_pred):
            if not clusters[y]:
                clusters[y] = {'data': []}
            clusters[y]['data'].append(data[i])

        if -1 in clusters:
            del clusters[-1]  # delete clusters classified as noise

        return clusters

    def _interpolate_clusters(self, clusters):
        """
        Fit a line equation onto each cluster
        :param clusters:
        :type clusters: dict A dictionary of clusters containing the detection data
        :return:
        """
        for cluster in clusters.iterkeys():
            m, c, r = self._get_line_equation(clusters[cluster]['data'])
            clusters[cluster] = {
                'm': m,
                'r': r,
                'c': c,
                'data': clusters[cluster]['data']
            }

        return clusters

    @staticmethod
    def _get_line_equation(data):
        """
        Return the gradient, correlation coefficient, and intercept from a give set of points
        :param data:
        :return:
        """
        warnings.filterwarnings('error')
        d = np.array(data)
        x = d[:, 0]
        y = d[:, 1]
        a = np.vstack([x, np.ones(len(x))]).T

        # determine gradient and y-intercept of data
        m, c = np.linalg.lstsq(a, y)[0]

        r = 0.0
        try:
            if np.round(c) > 0.0:
                # determine correlation coefficient if c > 0.0
                r = np.corrcoef(x, y)[0, 1]
        except Warning:
            r = 0.

        return m, c, r

    @staticmethod
    def _delete_clusters(clusters, clusters_to_delete):
        """
        Delete clusters with ids in clusters_to_delete
        :param clusters:
        :param clusters_to_delete:
        :return:
        """
        for c in clusters_to_delete:
            del clusters[c]
        return clusters

    def _merge_clusters(self, clusters):
        """
        Merge clusters based on how similar the gradient and y-intercept are
        :param clusters:
        :return:
        """

        while True:
            clusters_to_delete = []
            clusters_not_merged = 0
            for cluster_id in clusters.iterkeys():
                cluster1 = clusters[cluster_id]
                for cluster_id2 in clusters.iterkeys():
                    if cluster_id is cluster_id2:
                        continue

                    cluster2 = clusters[cluster_id2]
                    if self._clusters_are_similar(cluster1, cluster2):
                        cluster1['data'] = self._merge_clusters_data(cluster1, cluster2)
                        # recalculate equation of cluster
                        m, c, r = self._get_line_equation(cluster1['data'])
                        cluster1['m'] = m
                        cluster1['c'] = c
                        cluster1['r'] = r

                        cluster2['m'] = None  # mark cluster for deletion
                        clusters_to_delete.append(cluster_id2)
                    else:
                        clusters_not_merged += 1
            count = len(clusters) * (len(clusters) - 1)
            clusters = self._delete_clusters(clusters, clusters_to_delete)

            if clusters_not_merged >= count:
                break

        return clusters

    @staticmethod
    def _delete_dirty_clusters(clusters, threshold=0.85):
        """
        Delete clusters that have a low (< threshold) correlation coefficient
        :param clusters:
        :param threshold:
        :return:
        """
        good_clusters = {}
        for i, c in enumerate(clusters.iterkeys()):
            if abs(clusters[c]['r']) > threshold:
                good_clusters[i] = clusters[c]
        return good_clusters

    def detect(self, beam):
        log.debug('Running %s detection algorithm on beam %s', self.name, beam.id)
        if np.sum(beam.snr) == 0.0:
            log.debug('SNR is 0 for filtered beam %s', beam.id)
            return []

        db_scan_clusters = self._db_scan_cluster(beam.snr)
        clusters = self._interpolate_clusters(db_scan_clusters)
        clusters = self._delete_dirty_clusters(clusters, threshold=0.85)
        clusters = self._merge_clusters(clusters)

        # Visualise clusters
        for i, cluster in enumerate(clusters.iterkeys()):
            if clusters[cluster]:
                d = np.array(clusters[cluster]['data'])
                x = d[:, 0]
                y = d[:, 1]
                eq = str(i) + ') y = ' + str(round(clusters[cluster]['m'], 2)) + 'x + ' + str(
                    round(clusters[cluster]['c'], 2))
                eq += ' r = (' + str(round(clusters[cluster]['r'], 3)) + ')'
                if config.get_boolean('debug', 'DEBUG_CANDIDATES'):
                    plt.plot(x, y, 'o', label=eq)

        if config.get_boolean('debug', 'DEBUG_CANDIDATES') and clusters:
            plt.legend(loc='best', fancybox=True, framealpha=0.5)
            plt.xlabel('Channel')
            plt.title('Beam ' + str(beam.id))
            plt.ylabel('Time')
            plt.tight_layout()
            plt.grid()

            plt.show()

        candidates = []
        beam_candidates_counter = {}
        for cluster_id in clusters.iterkeys():
            cluster_data = clusters[cluster_id]['data']

            detection_data = np.array(
                [[beam.channels[c], beam.time[t], beam.snr[t][c]] for (t, c) in cluster_data])

            if beam.id not in beam_candidates_counter:
                beam_candidates_counter[beam.id] = 0
            beam_candidates_counter[beam.id] += 1

            candidate_name = str(beam.id) + '.' + str(beam_candidates_counter[beam.id])
            candidate = BeamSpaceDebrisCandidate(candidate_name, beam, detection_data)
            candidates.append(candidate)
        return candidates

    def _clusters_are_similar(self, cluster1, cluster2):
        """
        Determine whether two clusters are considered to be similar. In this case gradient and c-intercept similarity
        is used.
        :param cluster1:
        :param cluster2:
        :return:
        """
        # gradient
        m = self._is_similar(cluster1['m'], cluster2['m'], threshold=0.20)

        # intercept
        c = self._is_similar(cluster1['c'], cluster2['c'], threshold=0.20)

        return m and c

    def _is_similar(self, a, b, threshold=0.10):
        """
        Determine if two values are similar if they are within a certain threshold
        :param a:
        :param b:
        :param threshold:
        :return:
        """
        if a is None or b is None:
            return False

        percentage_difference = self._percentage_difference(a, b)

        if percentage_difference >= threshold:  # difference is more that 10 %
            return False

        return True

    @staticmethod
    def _percentage_difference(a, b):
        """
        Calculate the difference between two values
        :param a:
        :param b:
        :return:
        """
        diff = a - b
        mean = np.mean([a, b])
        percentage_difference = abs(diff / mean)

        return percentage_difference


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
        self.db_scan = DBSCAN(eps=self._eps, min_samples=self._min_samples, algorithm=self._algorithm)

    def detect(self, beam):
        clusters = self._create_clusters(beam)
        log.debug("%s detection clusters detected in beam %s", len(clusters), beam.id)

        merged_clusters = self._merge_clusters(clusters)
        log.debug("%s detection clusters remain after merging in beam %s", len(merged_clusters), beam.id)

        space_debris_candidates = self._create_space_debris_candidates(beam, merged_clusters)
        log.debug("%s space debris candidates detected", len(space_debris_candidates))

        return space_debris_candidates

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
            log.warning('DBSCAN failed. Beam data is empty')
            return []

        # Select only those labels which were not classified as noise (-1)
        filtered_cluster_labels = cluster_labels[cluster_labels > -1]

        # Group the data points in clusters
        clusters = []
        for label in np.unique(filtered_cluster_labels):
            cluster_data = data[np.where(cluster_labels == label)]

            try:
                cluster = DetectionCluster(cluster_data)

                # Add only those clusters that are linear
                if cluster.is_linear(threshold=0.9):
                    clusters.append(cluster)
            except ValueError:
                log.debug('Linear interpolation failed. No inliers found.')

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
                if cluster_1.is_cluster_similar(cluster_2, self._merge_threshold):
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

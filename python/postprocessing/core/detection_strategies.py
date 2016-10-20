import numpy as np
import matplotlib.pyplot as plt
import logging as log

from configuration.application import config
from abc import abstractmethod
from sklearn.cluster import DBSCAN
from detection_candidates import BeamSpaceDebrisCandidate


class SpaceDebrisDetection(object):
    def __init__(self, detection_strategy):
        self.detection_strategy = detection_strategy

    def detect(self, beam):
        candidates = self.detection_strategy.detect(beam)
        return candidates


class SpaceDebrisDetectionStrategy(object):
    def __init__(self):
        self.max_detections = 3

    @abstractmethod
    def detect(self, beam):
        pass


class DBScanSpaceDebrisDetectionStrategy(SpaceDebrisDetectionStrategy):
    name = 'DB Scan'

    def __init__(self):
        SpaceDebrisDetectionStrategy.__init__(self)
        pass

    @staticmethod
    def _merge_clusters_data(cluster1, cluster2):
        return cluster1['data'] + cluster2['data']

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
            clusters[cluster] = self._set_cluster_eq(clusters[cluster], m, c, r)

        return clusters

    @staticmethod
    def _set_cluster_eq(cluster, m, c, r):
        """
        Update the line equation of the cluster based on the passed parameters
        :param cluster:
        :param m: gradient
        :param c: intercept
        :param r: correlation
        :return:
        """
        cluster['m'] = m
        cluster['c'] = c
        cluster['r'] = r

        return cluster

    @staticmethod
    def _get_line_equation(data):
        """
        Return the gradient, correlation coefficient, and intercept from a give set of points
        :param data:
        :return:
        """
        d = np.array(data)
        x = d[:, 0]
        y = d[:, 1]
        a = np.vstack([x, np.ones(len(x))]).T

        # determine gradient and y-intercept of data
        m, c = np.linalg.lstsq(a, y)[0]

        r = 0.0
        if np.round(c) > 0.0:
            # determine correlation coefficient if c > 0.0
            r = np.corrcoef(x, y)[0, 1]

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
                        cluster1 = self._set_cluster_eq(cluster1, m, c, r)

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
                eq = 'y = ' + str(round(clusters[cluster]['m'], 3)) + 'x + ' + str(round(clusters[cluster]['c'], 3))
                eq += ' (' + str(round(clusters[cluster]['r'], 2)) + ')'
                if config.get_boolean('debug', 'DEBUG_CANDIDATES'):
                    plt.plot(x, y, 'o', label=eq)

        if config.get_boolean('debug', 'DEBUG_CANDIDATES'):
            plt.legend(loc='best', fancybox=True, framealpha=0.5)
            plt.xlabel('Channel')
            plt.ylabel('Time')
            plt.tight_layout()
            plt.grid()

            plt.show()

        candidates = []
        for cluster_id in clusters.iterkeys():
            cluster_data = clusters[cluster_id]['data']

            detection_data = np.array(
                [[beam.channels[c], beam.time[t], beam.snr[c][t]] for (c, t) in cluster_data])

            candidate = BeamSpaceDebrisCandidate(cluster_id, beam, detection_data)
            candidates.append(candidate)
        return candidates

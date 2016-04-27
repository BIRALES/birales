import numpy as np

from skimage.transform import hough_line
from skimage.transform import hough_line_peaks

from app.sandbox.postprocessing.lib.SpaceDebrisDetection import SpaceDebrisDetectionStrategy
from app.sandbox.postprocessing.models.SpaceDebrisCandidate import SpaceDebrisCandidate

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import app.sandbox.postprocessing.config.application as config


class DBScanSpaceDebrisDetectionStrategy(SpaceDebrisDetectionStrategy):
    def __init__(self, max_detections):
        SpaceDebrisDetectionStrategy.__init__(self, max_detections)
        pass

    @staticmethod
    def merge_clusters_data(cluster1, cluster2):
        return cluster1['data'] + cluster2['data']

    def clusters_are_similar(self, cluster1, cluster2):
        """
        Determine whether two clusters are considered to be similar. In this case gradient and c-intercept similarity
        is used.
        :param cluster1:
        :param cluster2:
        :return:
        """
        # gradient
        m = self.is_similar(cluster1['m'], cluster2['m'], threshold = 0.20)

        # intercept
        c = self.is_similar(cluster1['c'], cluster2['c'], threshold = 0.20)

        return m and c

    def is_similar(self, a, b, threshold = 0.10):
        """
        Determine if two values are similar if they are within a certain threshold
        :param a:
        :param b:
        :param threshold:
        :return:
        """
        if a is None or b is None:
            return False

        percentage_difference = self.percentage_difference(a, b)

        if percentage_difference >= threshold:  # difference is more that 10 %
            return False

        return True

    @staticmethod
    def percentage_difference(a, b):
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
    def db_scan_cluster(data):
        """
        Use the DBScan algorithm to create a set of clusters from the given beam data
        :param data:
        :return:
        """
        data = np.transpose(np.nonzero(data > 0.))
        y_pred = DBSCAN(eps = 10.0, min_samples = 5, algorithm = 'kd_tree').fit_predict(data)
        clusters = dict.fromkeys(np.unique(y_pred))

        # Classify coordinates into corresponding clusters
        for i, y in enumerate(y_pred):
            if not clusters[y]:
                clusters[y] = {'data': []}
            clusters[y]['data'].append(data[i])

        del clusters[-1]  # delete clusters classified as noise

        return clusters

    def interpolate_clusters(self, clusters):
        """
        Fit a line equation onto each cluster
        :param clusters:
        :return:
        """
        for cluster in clusters.iterkeys():
            m, c, r = self.get_line_equation(clusters[cluster]['data'])
            clusters[cluster] = self.set_cluster_eq(clusters[cluster], m, c, r)

        return clusters

    @staticmethod
    def set_cluster_eq(cluster, m, c, r):
        """
        Update the line equation of the cluster based on the passed parameters
        :param cluster:
        :param m:
        :param c:
        :param r:
        :return:
        """
        cluster['m'] = m
        cluster['c'] = c
        cluster['r'] = r

        return cluster

    @staticmethod
    def get_line_equation(data):
        """
        Return the gradient, correlation coefficient, and intercept from a give set of points
        :param data:
        :return:
        """
        d = np.array(data)
        x = d[:, 0]
        y = d[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T

        m, c = np.linalg.lstsq(A, y)[0]  # determine gradient and y-intercept of data
        r = np.corrcoef(x, y)[0, 1]  # determine correlation coefficient
        return m, c, r

    @staticmethod
    def delete_clusters(clusters, clusters_to_delete):
        """
        Delete clusters with ids in clusters_to_delete
        :param clusters:
        :param clusters_to_delete:
        :return:
        """
        for c in clusters_to_delete:
            del clusters[c]
        return clusters

    def merge_clusters(self, clusters):
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
                    if self.clusters_are_similar(cluster1, cluster2):
                        cluster1['data'] = self.merge_clusters_data(cluster1, cluster2)
                        # recalculate equation of cluster
                        m, c, r = self.get_line_equation(cluster1['data'])
                        cluster1 = self.set_cluster_eq(cluster1, m, c, r)

                        cluster2['m'] = None  # mark cluster for deletion
                        clusters_to_delete.append(cluster_id2)
                    else:
                        clusters_not_merged += 1
            count = len(clusters) * (len(clusters) - 1)
            clusters = self.delete_clusters(clusters, clusters_to_delete)

            if clusters_not_merged >= count:
                break

        return clusters

    @staticmethod
    def delete_dirty_clusters(clusters, threshold = 0.85):
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
        db_scan_clusters = self.db_scan_cluster(beam.data.snr)
        clusters = self.interpolate_clusters(db_scan_clusters)
        clusters = self.delete_dirty_clusters(clusters, threshold = 0.85)
        clusters = self.merge_clusters(clusters)

        # Visualise clusters
        for i, cluster in enumerate(clusters.iterkeys()):
            if clusters[cluster]:
                d = np.array(clusters[cluster]['data'])
                x = d[:, 0]
                y = d[:, 1]
                eq = 'y = ' + str(round(clusters[cluster]['m'], 3)) + 'x + ' + str(round(clusters[cluster]['c'], 3))
                eq += ' (' + str(round(clusters[cluster]['r'], 2)) + ')'
                if config.DEBUG_CANDIDATES:
                    plt.plot(x, y, 'o', label = eq)

        if config.DEBUG_CANDIDATES:
            plt.style.use('ggplot')
            plt.legend(loc = 'best', fancybox = True, framealpha = 0.5)
            plt.xlabel('Channel')
            plt.ylabel('Time')
            plt.tight_layout()
            plt.grid()
            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())
            plt.show()
            exit(0)

        candidates = []
        for cluster_id in clusters.iterkeys():
            cluster_data = clusters[cluster_id]['data']

            detection_data = np.array(
                    [[beam.data.channels[c], beam.data.time[t], beam.data.snr[c][t]] for (c, t) in cluster_data])
            candidate = SpaceDebrisCandidate(tx = 100, beam = beam, detection_data = detection_data)
            candidates.append(candidate)

        return candidates

    def detect_km(self, beam):
        random_state = 170
        data = beam.data.snr
        data = np.transpose(np.nonzero(data > 2.))
        h, theta, d = hough_line(beam.data.snr)
        h_space, angles, dists = hough_line_peaks(h, theta, d, 10)

        n_clusters = len(angles)
        k_means = KMeans(n_clusters = n_clusters, random_state = random_state)
        y_pred = k_means.fit_predict(data)

        # import matplotlib.pyplot as plt
        # plt.scatter(data[:, 0], data[:, 1], c = y_pred)
        # plt.show()
        # exit(0)

        # Populate cluster with coordinates
        clusters = dict.fromkeys(np.unique(y_pred))
        for i, y in enumerate(y_pred):
            if not clusters[y]:
                clusters[y] = []
            clusters[y].append(data[i])

        candidates = []
        for cluster in clusters.iterkeys():
            detection_data = []
            for detection_index, (channel, time) in enumerate(clusters[cluster]):
                time = beam.data.time[time]
                channel = beam.data.channels[channel]
                snr = beam.data.snr[channel][time]
                detection_data.append([channel, time, snr])
            if detection_data:
                candidate = SpaceDebrisCandidate(tx = 100, beam = beam, detection_data = np.array(detection_data))
                candidates.append(candidate)

        print len(candidates), 'candidates were detected'
        return candidates

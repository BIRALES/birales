import logging as log
import matplotlib.pyplot as plt
import numpy as np
import warnings

from abc import abstractmethod
from sklearn.cluster import DBSCAN
from postprocessing.configuration.application import config
from detection_candidates import BeamSpaceDebrisCandidate
from sklearn import linear_model


class SpaceDebrisDetection(object):
    def __init__(self, detection_strategy):
        self.detection_strategy = detection_strategy

    def detect(self, beam):
        candidates = self.detection_strategy.detect(beam)
        return candidates


class SpaceDebrisDetectionStrategy(object):
    def __init__(self):
        self.max_detections = config.get_int('application', 'MAX_DETECTIONS')

    @abstractmethod
    def detect(self, beam):
        pass


class SpiritSpaceDebrisDetectionStrategy(SpaceDebrisDetectionStrategy):
    name = 'Spirit'

    def __init__(self):
        SpaceDebrisDetectionStrategy.__init__(self)
        pass

    def detect(self, beam):
        clusters = self._create_clusters(beam)
        processed_clusters = self._process_clusters(clusters)
        space_derbis_candidates = self._create_space_debris_candidates(processed_clusters)

        return []

    def _create_clusters(self, beam):
        """
        Use the DBScan algorithm to create a set of clusters from the given beam data
        :param beam: The beam object from which the clusters will be generated
        :return:
        """

        # Initialise clustering algorithm
        db_scan = DBSCAN(eps=10.0, min_samples=5, algorithm='kd_tree')

        # Select the data points that are non-zero and transform them in a time (x), channel (y) nd-array
        data = np.column_stack(np.where(beam.snr > 0.))

        # Perform clustering on the data and returns cluster labels the points are associated with
        cluster_labels = db_scan.fit_predict(data)

        # Select only those labels which were not classified as noise (-1)
        filtered_cluster_labels = cluster_labels[cluster_labels > -1]

        # Group the data points in clusters
        clusters = dict.fromkeys(np.unique(filtered_cluster_labels))
        for label in filtered_cluster_labels:
            clusters[label] = data[np.where(cluster_labels == label)]

        return clusters

    @staticmethod
    def _linear_model(dirty_clusters):
        clusters = {}
        for label in dirty_clusters:
            clusters[label] = {}
            # Fit line using all data
            x = dirty_clusters[label][:, [1]]
            y = dirty_clusters[label][:, 0]

            # Robustly fit linear model with RANSAC algorithm
            model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
            model_ransac.fit(x, y)

            # Remove outliers - select data points that are in-liers
            inlier_mask = model_ransac.inlier_mask_

            if model_ransac.estimator_.score(x, y) < 0.90:
                continue

            clusters[label]['data'] = dirty_clusters[label][inlier_mask]

            # Gradient
            clusters[label]['m'] = model_ransac.estimator_.coef_[0]

            # Intercept
            clusters[label]['c'] = model_ransac.estimator_.intercept_

            # The coefficient of determination R^2 of the prediction.
            clusters[label]['r'] = model_ransac.estimator_.score(x, y)

        return clusters


    def _process_clusters(self, dirty_clusters):
        clusters = self._linear_model(dirty_clusters)
        if clusters[0]:
            print 'cluster'
            return clusters
        pass

    def _create_space_debris_candidates(self, clusters):
        pass

    def _visualise_cluster(self, cluster):
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
            # m, c, r = self._ransac(clusters[cluster]['data'])
            clusters[cluster] = {
                'm': m,
                'r': r,
                'c': c,
                'data': clusters[cluster]['data']
            }

        return clusters

    def _ransac(self, data):
        from sklearn import linear_model
        d = np.array(data)
        x = d[:, [0]]
        y = d[:, 1]
        linear_model = linear_model.LinearRegression()
        linear_model.fit(x, y)

        c = linear_model.intercept_
        m = linear_model.coef_

        pass

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
        log.debug('Running DBSCAN space debris detection algorithm on beam %s', beam.id)

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

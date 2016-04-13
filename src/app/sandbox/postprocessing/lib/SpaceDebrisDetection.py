import numpy as np
from abc import abstractmethod

from skimage.transform import hough_line
from skimage.transform import hough_line_peaks

from app.sandbox.postprocessing.models.SpaceDebrisCandidate import SpaceDebrisCandidate
from app.sandbox.postprocessing.helpers.LineGeneratorHelper import LineGeneratorHelper
from app.sandbox.postprocessing.lib.SpaceDebrisCandidateCollection import SpaceDebrisCandidateCollection

import app.sandbox.postprocessing.config.application as app_config
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics

import matplotlib.pyplot as plt


class SpaceDebrisDetection(object):
    def __init__(self, detection_strategy):
        self.detection_strategy = detection_strategy

    def detect(self, beam):
        detections = self.detection_strategy.detect(beam)
        candidates = SpaceDebrisCandidateCollection(candidates = detections)
        return candidates


class SpaceDebrisDetectionStrategy(object):
    def __init__(self, max_detections = 10):
        self.max_detections = max_detections  # maximum detections per beam

    @abstractmethod
    def detect(self, beam):
        pass


class KMeansSpaceDebrisDetectionStrategy(SpaceDebrisDetectionStrategy):
    def __init__(self, max_detections):
        SpaceDebrisDetectionStrategy.__init__(self, max_detections)
        pass

    @staticmethod
    def merge_clusters_data(cluster1, cluster2):
        return cluster1['data'] + cluster2['data']

    def clusters_are_similar(self, cluster1, cluster2):
        # gradient
        m = self.is_similar(cluster1['m'], cluster2['m'], threshold = 0.10)

        # intercept
        c = self.is_similar(cluster1['c'], cluster2['c'], threshold = 0.20)

        return m and c

    def is_similar(self, a, b, threshold = 0.10):
        if a is None or b is None:
            return False

        percentage_difference = self.percentage_difference(a, b)

        if percentage_difference >= threshold:  # difference is more that 10 %
            return False

        return True

    @staticmethod
    def percentage_difference(a, b):
        diff = a - b
        mean = np.mean([a, b])
        percentage_difference = abs(diff / mean)

        return percentage_difference

    @staticmethod
    def db_scan_cluster(data):
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

    @staticmethod
    def interpolate_clusters(clusters):
        """
        Fit a line
        :param clusters:
        :return:
        """
        for cluster in clusters.iterkeys():
            d = np.array(clusters[cluster]['data'])
            x = d[:, 0]
            y = d[:, 1]
            A = np.vstack([x, np.ones(len(x))]).T

            m, c = np.linalg.lstsq(A, y)[0]  # determine gradient and y-intercept of data
            r = np.corrcoef(x, y)[0, 1]  # determine correlation coefficient

            clusters[cluster]['m'] = m
            clusters[cluster]['c'] = c
            clusters[cluster]['r'] = r

        return clusters

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
                plt.plot(x, y, 'o', label = eq)
        plt.legend()
        plt.show()
        exit(0)

    def detect_km(self, beam):
        random_state = 170
        data = beam.data.snr
        data = np.transpose(np.nonzero(data > 2.))
        h, theta, d = hough_line(beam.data.snr)
        h_space, angles, dists = hough_line_peaks(h, theta, d, 10)
        print np.rad2deg(angles)
        n_clusters = len(angles)
        k_means = KMeans(n_clusters = n_clusters, random_state = random_state)
        y_pred = k_means.fit_predict(data)
        # print len(y_pred)
        # import matplotlib.pyplot as plt
        # plt.scatter(data[:, 0], data[:, 1], c = y_pred)
        # plt.show()
        # exit(0)
        # print data.shape
        # print np.unique(y_pred)
        # for i, (x, y) in enumerate(data):
        #     print i, x, y, beam.data.snr[x, y]
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

    def get_number_of_clusters(self):
        pass

    def get_cluster_center(self, coordinates):
        return int(len(coordinates) / 2)


class LineSpaceDebrisDetectionStrategy(SpaceDebrisDetectionStrategy):
    def __init__(self, max_detections):
        SpaceDebrisDetectionStrategy.__init__(self, max_detections)
        self.snr_threshold = app_config.SNR_DETECTION_THRESHOLD
        self.hough_threshold = 10

    # todo refactor this function
    def detect(self, beam):
        hough_lines_coordinates = self.hough_transform(beam.data)

        candidates = []  # todo replace with candidate collection
        max_channel = beam.data.get_max_channel()
        min_channel = 0
        max_time = beam.data.get_max_time()
        min_time = 0
        for detection_index, ((f0, t0), (fn, tn)) in enumerate(hough_lines_coordinates):
            discrete_h_line = LineGeneratorHelper.get_line((f0, t0), (fn, tn))

            detection_data = []
            for coordinate in discrete_h_line:
                channel = coordinate[0]  # not sure why this works
                time = coordinate[1]  # not sure why this works

                if min_channel < channel < max_channel and min_time < time < max_time:
                    snr = beam.data.snr[channel][time]
                    if snr >= self.snr_threshold:  # todo change this to a more appropriate detection metric
                        time = beam.data.time[time]
                        channel = beam.data.channels[channel]
                        detection_data.append([channel, time, snr])

            if detection_data:
                candidate = SpaceDebrisCandidate(tx = 100, beam = beam, detection_data = detection_data)
                candidates.append(candidate)

        # todo replace with log
        print len(candidates), 'candidates were detected'

        return candidates

    def hough_transform(self, beam_data):
        h, theta, d = hough_line(beam_data.snr)

        # self.visualise_hough_space(h, theta, d)

        t0 = 0
        tn = beam_data.get_max_time()
        hough_line_coordinates = []
        h_space, angles, dists = hough_line_peaks(h, theta, d, self.hough_threshold)
        for _, angle, dist in zip(h_space, angles, dists):
            f0 = (dist - t0 * np.cos(angle)) / np.sin(angle)
            fn = (dist - tn * np.cos(angle)) / np.sin(angle)

            coordinate = ((f0, t0), (fn, tn))
            hough_line_coordinates.append(coordinate)

        return hough_line_coordinates

    @staticmethod
    def visualise_hough_space(h, theta, d):
        import matplotlib.pyplot as plt
        plt.imshow(np.log(1 + h),
                   extent = [np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
                             d[-1], d[0]],
                   cmap = plt.cm.gray, aspect = 1 / 1.5)

        # plt.plot(h)
        plt.show()

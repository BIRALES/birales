import numpy as np
from abc import abstractmethod

from skimage.transform import hough_line
from skimage.transform import hough_line_peaks

from app.sandbox.postprocessing.models.SpaceDebrisCandidate import SpaceDebrisCandidate
from app.sandbox.postprocessing.helpers.LineGeneratorHelper import LineGeneratorHelper
from app.sandbox.postprocessing.lib.SpaceDebrisCandidateCollection import SpaceDebrisCandidateCollection

import app.sandbox.postprocessing.config.application as app_config
from sklearn.cluster import KMeans


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

    def detect(self, beam):
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

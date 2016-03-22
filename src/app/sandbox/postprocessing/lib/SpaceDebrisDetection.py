import numpy as np
from abc import abstractmethod

from skimage.transform import hough_line
from skimage.transform import hough_line_peaks

from app.sandbox.postprocessing.models.SpaceDebrisCandidate import SpaceDebrisCandidate
from app.sandbox.postprocessing.helpers.LineGeneratorHelper import LineGeneratorHelper


class SpaceDebrisDetection(object):
    def __init__(self, detection_strategy):
        self.detection_strategy = detection_strategy

    def detect(self, beam_id, beam_data):
        detections = self.detection_strategy.detect(beam_id, beam_data)
        return detections


class SpaceDebrisDetectionStrategy(object):
    def __init__(self, max_detections):
        self.max_detections = max_detections  # maximum detections per beam

    @abstractmethod
    def detect(self, beam_id, beam_data):
        pass


class LineSpaceDebrisDetectionStrategy(SpaceDebrisDetectionStrategy):
    def __init__(self, max_detections):
        SpaceDebrisDetectionStrategy.__init__(self, max_detections)
        self.snr_threshold = 1.0  # the snr power at which a detection is determined
        self.hough_threshold = 10

    def detect(self, beam_id, beam_data):
        hough_lines_coordinates = self.hough_transform(beam_data)

        candidates = []
        max_frequency = beam_data.frequencies[-1]
        min_frequency = beam_data.frequencies[0]
        for detection_index, ((f0, t0), (fn, tn)) in enumerate(hough_lines_coordinates):
            print 'Draw Hough Line at', (f0, t0), (fn, tn)
            discrete_h_line = LineGeneratorHelper.get_line((f0, t0), (fn, tn))

            detection_data = []
            for coordinate in discrete_h_line:
                frequency = coordinate[0] + 1.  # not sure why this works
                time = coordinate[1] + 1.  # not sure why this works

                if min_frequency < frequency < max_frequency and 0 < time < beam_data.time:
                    snr = beam_data.snr[frequency][time]
                    # if snr == self.snr_threshold:
                    detection_data.append([frequency, time, snr])

            candidate = SpaceDebrisCandidate(100, beam_id, detection_data)
            candidates.append(candidate)
        return candidates

    def hough_transform(self, beam_data):
        h, theta, d = hough_line(beam_data.snr)
        # self.visualise_hough_space(h,theta,d)
        t0 = 0
        tn = beam_data.time
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

        plt.plot(h)
        plt.show()

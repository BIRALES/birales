import numpy as np
from abc import abstractmethod

from skimage.transform import hough_line
from skimage.transform import hough_line_peaks

from app.sandbox.postprocessing.models.SpaceDebrisCandidate import SpaceDebrisCandidate
from app.sandbox.postprocessing.helpers.LineGeneratorHelper import LineGeneratorHelper
from app.sandbox.postprocessing.lib.SpaceDebrisCandidateCollection import SpaceDebrisCandidateCollection


class SpaceDebrisDetection(object):
    def __init__(self, detection_strategy):
        self.detection_strategy = detection_strategy

    def detect(self, beam):
        detections = self.detection_strategy.detect(beam)
        candidates = SpaceDebrisCandidateCollection(candidates = detections)
        return candidates


class SpaceDebrisDetectionStrategy(object):
    def __init__(self, max_detections):
        self.max_detections = max_detections  # maximum detections per beam

    @abstractmethod
    def detect(self, beam):
        pass


class LineSpaceDebrisDetectionStrategy(SpaceDebrisDetectionStrategy):
    def __init__(self, max_detections):
        SpaceDebrisDetectionStrategy.__init__(self, max_detections)
        # todo replace these from configuration
        self.snr_threshold = 1.0  # the snr power at which a detection is determined
        self.hough_threshold = 10

    # todo refactor this function
    def detect(self, beam):
        hough_lines_coordinates = self.hough_transform(beam.data)

        candidates = []  # todo replace with candidate collection
        max_channel = beam.data.get_max_channel()
        min_channel = beam.data.get_min_channel()
        max_time = beam.data.get_max_time()
        min_time = beam.data.get_min_time()
        for detection_index, ((f0, t0), (fn, tn)) in enumerate(hough_lines_coordinates):
            discrete_h_line = LineGeneratorHelper.get_line((f0, t0), (fn, tn))

            detection_data = []
            for coordinate in discrete_h_line:
                channel = coordinate[0] # + 1.  # not sure why this works
                time = coordinate[1] # + 1.  # not sure why this works

                if min_channel < channel < max_channel and min_time < time < max_time:
                    snr = beam.data.snr[channel][time]
                    if snr > self.snr_threshold:  # todo change this to a more appropriate detection metric
                        time = beam.data.time[time]
                        channel = beam.data.channels[channel]
                        detection_data.append([channel, time, snr])

            if detection_data:
                candidate = SpaceDebrisCandidate(tx = 100, beam = beam, detection_data = detection_data)
                candidates.append(candidate)

        print len(candidates), 'candidates were detected'

        return candidates

    def hough_transform(self, beam_data):
        h, theta, d = hough_line(beam_data.snr)

        # self.visualise_hough_space(h, theta, d)

        t0 = beam_data.get_min_time()
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

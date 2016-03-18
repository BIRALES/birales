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

    def detect(self, beam_id, beam_data):
        hough_lines_coordinates = self.hough_transform(beam_data)

        candidates = []
        max_frequency = beam_data.frequencies[-1]
        for detection_index, ((t0, f0), (tn, fn)) in enumerate(hough_lines_coordinates):
            discrete_h_line = LineGeneratorHelper.get_line((t0, f0), (tn, fn))
            detection_data = []
            for coordinate in discrete_h_line:
                time = coordinate[0] + 1.  # not sure why this works
                frequency = coordinate[1] + 1.  # not sure why this works

                if frequency < max_frequency:
                    snr = beam_data.snr[frequency][time]
                    if snr == self.snr_threshold:
                        detection_data.append([frequency, time, snr])

            candidate = SpaceDebrisCandidate(100, beam_id, detection_data)
            candidates.append(candidate)
        return candidates

    def hough_transform(self, beam_data):
        h, theta, d = hough_line(beam_data.snr)

        t0 = 0
        tn = beam_data.time
        time = []
        freq = []
        hough_line_coordinates = []
        h_space, angles, dists = hough_line_peaks(h, theta, d, num_peaks = self.max_detections)

        for _, angle, dist in zip(h_space, angles, dists):
            f0 = (dist - t0 * np.cos(angle)) / np.sin(angle)
            fn = (dist - tn * np.cos(angle)) / np.sin(angle)

            freq.append((f0, fn))
            time.append((t0, tn))

            coordinate = ((t0, f0), (tn, fn))
            hough_line_coordinates.append(coordinate)

        return hough_line_coordinates

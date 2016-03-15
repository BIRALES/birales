import numpy as np
from skimage.transform import hough_line
from skimage.transform import hough_line_peaks
from app.sandbox.postprocessing.helpers.LineGeneratorHelper import LineGeneratorHelper
from app.sandbox.postprocessing.helpers.DateTimeHelper import DateTimeHelper


class SpaceDebrisDetector:
    def __init__(self, max_detections):
        self.max_detections = max_detections  # maximum detections per beam
        pass

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

        print 'There were', len(angles), 'detections'

        return hough_line_coordinates

    def get_detections(self, beam_data):
        # todo - this can be handled better; consider refactoring using a strategy design pattern
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
                    if snr == 1.:  # todo - replace with and value greater than threshold
                        detection_data.append([frequency, time, snr])

            candidate = SpaceDebrisCandidate(100, beam_data.id, detection_data)
            candidates.append(candidate)
        return candidates


class SpaceDebrisCandidate:
    def __init__(self, tx, beam_id, detection_data):
        self.beam_id = beam_id
        self.tx = tx

        self.data = {
            'time': [],
        }

        self.populate(detection_data)

    def populate(self, detection_data):
        for frequency, time, snr in detection_data:
            self.data['time'] = time
            self.data['mdj2000'] = SpaceDebrisCandidate.get_mdj2000(time)
            self.data['time_elapsed'] = SpaceDebrisCandidate.time_elapsed(time)
            self.data['doppler_shift'] = SpaceDebrisCandidate.get_doppler_shift(self.tx, frequency),
            self.data['snr'] = snr

    @staticmethod
    def get_doppler_shift(tf, frequency):
        return frequency - tf

    @staticmethod
    def time_elapsed(time):
        return time

    @staticmethod
    def get_mdj2000(dates):
        return DateTimeHelper.ephem2juldate(dates)

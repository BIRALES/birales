import numpy as np
from skimage.transform import hough_line
from skimage.transform import hough_line_peaks

from app.sandbox.postprocessing.helpers.DateTimeHelper import DateTimeHelper
from app.sandbox.postprocessing.helpers.LineGeneratorHelper import LineGeneratorHelper
from app.sandbox.postprocessing.helpers.TableMakerHelper import TableMakerHelper
from app.sandbox.postprocessing.views.BeamDataView import BeamDataView


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

        return hough_line_coordinates

    def get_detections(self, beam_id, beam_data):
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

            candidate = SpaceDebrisCandidate(100, beam_id, detection_data)
            candidates.append(candidate)
        return candidates


class SpaceDebrisCandidate:
    def __init__(self, tx, beam_id, detection_data):
        self.beam_id = beam_id
        self.tx = tx

        self.data = {
            'time': [],
            'mdj2000': [],
            'time_elapsed': [],
            'frequency': [],
            'doppler_shift': [],
            'snr': [],
        }

        self.populate(detection_data)

    def populate(self, detection_data):
        for frequency, time, snr in detection_data:
            self.data['time'].append(time)
            self.data['mdj2000'].append(SpaceDebrisCandidate.get_mdj2000(time))
            self.data['time_elapsed'].append(SpaceDebrisCandidate.time_elapsed(time))
            self.data['frequency'].append(frequency)
            self.data['doppler_shift'].append(SpaceDebrisCandidate.get_doppler_shift(self.tx, frequency))
            self.data['snr'].append(snr)

    def view(self, beam_data, name = 'Superimposed candidate'):
        view = BeamDataView(name)
        view.set_layout(figure_title = name,
                        x_axis_title = 'Frequency (Hz)',
                        y_axis_title = 'Time (s)')

        for time, frequency in zip(self.data['time'], self.data['frequency']):
            snr = beam_data.snr[frequency][time]
            if snr == 1.:
                beam_data.snr[frequency][time] = 2.0

        view.set_data(data = beam_data.snr,
                      x_axis = beam_data.time,
                      y_axis = beam_data.frequencies
                      )
        view.show()

    def table_view(self, name = 'Input'):
        table = TableMakerHelper()
        table.set_headers([
            'Epoch',
            'MJD2000',
            'Time Delay',
            'Frequency',
            'Doppler',
            'SNR'
        ])

        table.set_rows(self.data)
        table.visualise(name)

    @staticmethod
    def get_doppler_shift(tf, frequency):
        return frequency - tf

    @staticmethod
    def time_elapsed(time):
        return time

    @staticmethod
    def get_mdj2000(dates):
        return DateTimeHelper.ephem2juldate(dates)

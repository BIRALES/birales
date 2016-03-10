import numpy as np
import csv
from plotly.offline import plot
import plotly.graph_objs as go
from skimage.transform import hough_line
from skimage.transform import hough_line_peaks
from scipy import signal
from helpers import TableMakerHelper
from helpers import DateTimeHelper
from BeamData import BeamData

from skimage.morphology import disk
from skimage.filters.rank import median
from scipy import ndimage
import matplotlib.pyplot as plt


class OrbitDeterminationInputGenerator:
    mock_data_dir = 'data/'
    output_dir = 'output/'
    max_detections = 3

    def __init__(self):
        self.max_time = 600
        self.min_time = 0
        return

    def mock_data(self, file_name):
        with open(self.mock_data_dir + file_name) as csv_file:
            row = csv.reader(csv_file)
            mock_data = list(row)

        return mock_data

    def visualise_hough_transform(self, name, hough_transform_data):
        data = [
            go.Heatmap(
                z = hough_transform_data,
                colorscale = 'Viridis',
                # colorbar = {'title': 'Hough Transform'},
            )
        ]

        layout = go.Layout(
            title = 'Beam Data',
            xaxis = dict(ticks = '', nticks = 36, title = 'Time'),
            yaxis = dict(ticks = '', title = 'Frequency'),

        )

        fig = go.Figure(data = data, layout = layout)

        plot(fig, filename = self.output_dir + name + '.html')

    def visualise_mock_data(self):
        table = TableMakerHelper()
        table.set_headers([
            'Epoch',
            'MJD2000',
            'Time Delay',
            'Doppler',
            'SNR'
        ])

        mock_data = self.mock_data('beam_sample_output.csv')
        table.set_rows(mock_data)

        table.visualise_with_graph('Beam_4')

    def visualise_line(self, name, image, lines_xs = None, lines_ys = None):
        traces = []
        if lines_xs and lines_ys:
            for x, y in zip(lines_xs, lines_ys):
                traces.append(go.Scatter(
                    x = x,
                    y = y,
                    name = 'Line',
                    connectgaps = True
                ))

        traces.append(
            go.Heatmap(
                z = image,
                colorscale = 'Viridis',
            )
        )

        layout = go.Layout(
            title = 'Beam Data',
            xaxis = dict(ticks = '', nticks = 36, title = 'Time'),
            yaxis = dict(ticks = '', title = 'Frequency'),

        )

        fig = go.Figure(data = traces, layout = layout)

        plot(fig, filename = self.output_dir + name + '.html')

    def hough_transform(self, image, visualise = False):
        h, theta, d = hough_line(image)

        if visualise:
            hough_transform_data = np.log(1 + h)
            self.visualise_hough_transform('Hough Tansform', hough_transform_data)

        x0 = self.min_time
        x1 = self.max_time
        x = []
        y = []

        h_space, angles, dists = hough_line_peaks(h, theta, d, num_peaks = self.max_detections)

        for _, angle, dist in zip(h_space, angles, dists):
            y0 = (dist - x0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)

            x.append([x0, x1])
            y.append([y0, y1])
            # m = np.gradient(y, x)
            # print 'y=', m, 'x +', y0

        print 'There were', len(angles), 'detections'

        return x, y

    def remove_noise(self, noisy_data):
        # Remove instaces that are 3 stds away from the mean
        noisy_data[noisy_data < np.mean(noisy_data) + 5. * np.std(noisy_data)] = 0.

        print 'Mean', np.mean(noisy_data)
        print 'STD', np.std(noisy_data)

        return noisy_data

    def line_detection(self, beam_data, visualise = False):
        power = beam_data.power

        image = np.array(power)
        image = self.remove_noise(image)

        lines_x, lines_y = self.hough_transform(image = image)

        self.visualise_line('Line', power)
        self.visualise_line('Line Detection', image, lines_x, lines_y)

        return

    def get_line_coordinates(self, line_x, line_y, image):

        m = np.gradient(line_y, line_x)
        for x in range(self.min_time, self.max_time):
            y = m * x + line_y
            print x, y

od = OrbitDeterminationInputGenerator()
bd = BeamData()

# bd.visualise(power, time, freq, 'Mock_BeamData')

hough = od.line_detection(bd)

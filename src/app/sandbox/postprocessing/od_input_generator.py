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
    max_frequency = 200

    def __init__(self):
        self.max_time = 600
        self.min_time = 0
        return

    @staticmethod
    def get_line(start, end):
        """Bresenham's Line Algorithm
        """

        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)

        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True

        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1

        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx

        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()
        return points

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
                    mode = 'lines+markers',
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
        xs = []
        ys = []

        h_space, angles, dists = hough_line_peaks(h, theta, d, num_peaks = self.max_detections)

        for _, angle, dist in zip(h_space, angles, dists):
            y0 = (dist - x0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)

            c = y0
            m = self.gradient(y0, y1, x0, x1)

            # print y0, y1
            # y = []
            # for x in range(0, self.max_frequency):
            #     yy = m * x + c
            #
            #     y.append(m * x + c)
            ys.append((y0, y1))
            xs.append((x0, x1))
            print angle
            print 'y=', m, 'x +', y0

        print 'There were', len(angles), 'detections'

        return xs, ys

    def gradient(self, y0, y1, x0, x1):
        return (y1 - y0) / (x1 - x0)

    def remove_noise(self, noisy_data):
        # Remove instaces that are 3 stds away from the mean
        noisy_data[noisy_data < np.mean(noisy_data) + 5. * np.std(noisy_data)] = 0.

        return noisy_data

    def line_detection(self, beam_data, visualise = False):
        power = beam_data.power

        image = np.array(power)
        image = self.remove_noise(image)

        lines_x, lines_y = self.hough_transform(image = image)

        # self.visualise_line('Original', image)
        image = self.print_line_on_basemap(image, lines_x, lines_y)
        # self.visualise_line('With Hough Line', image)
        return

    def print_line_on_basemap(self, image, lines_time, lines_frequencies):
        x0 = lines_time[0][0]
        x1 = lines_time[0][-1]

        y0 = lines_frequencies[0][0]
        y1 = lines_frequencies[0][-1]

        print x0, y0, x1, y1

        line_coor = BeamData.get_line((x0, y0), (x1, y1))  # (x0, y0), (x1, y1)
        hits = 0
        for coordinate in line_coor:
            x = coordinate[0] + 1  # not sure why this works
            y = coordinate[1] + 1  # not sure why this works

            # print x, y
            if y < len(image):
                if image[y][x] == 1.:
                    image[y][x] = 3.0  # hit
                    hits += 1
                else:
                    image[y][x] = 0.0  # paint line
        print 'There were', hits, 'hits'
        return image

    def print_line_as_heatmap(self, image, xs, ys):
        z = []
        xs = xs[0]

        # print ys[0][0:10]
        print np.nonzero(image[170])  # y=170, x= 171
        print 'ys', np.round(ys[0])[170:180].tolist()
        print 'xs', xs[170:180]
        ys = np.round(ys[0])

        print len(image[0]), len(image), len(ys), len(xs)

        # print xs[0:10]

        for x in range(0, self.max_frequency):
            y = int(ys[x])
            if y >= 0:
                image[y][x] = 2.0
                # print x, y
        return image


od = OrbitDeterminationInputGenerator()
bd = BeamData()

# bd.visualise(power, time, freq, 'Mock_BeamData')

hough = od.line_detection(bd)

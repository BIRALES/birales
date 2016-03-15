import numpy as np
import csv
from plotly.offline import plot
import plotly.graph_objs as go
from skimage.transform import hough_line
from skimage.transform import hough_line_peaks
from helpers import TableMakerHelper
from helpers import LineGeneratorHelper
from BeamData import BeamData


class OrbitDeterminationInputGenerator:
    mock_data_dir = 'results/'
    data_dir = 'results/'
    output_dir = 'output/'
    max_detections = 3
    max_frequency = 200

    def __init__(self):
        self.max_time = 600
        self.min_time = 0
        return

    def mock_result(self, file_name):
        with open(self.mock_data_dir + file_name) as csv_file:
            row = csv.reader(csv_file)
            mock_data = list(row)

        return mock_data

    def visualise_mock_result(self):
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

    def visualise_hough_transform(self, name, hough_transform_data):
        data = [
            go.Heatmap(
                z = hough_transform_data,
                colorscale = 'Viridis',
            )
        ]

        layout = go.Layout(
            title = 'Beam Data',
            xaxis = dict(ticks = '', nticks = 36, title = 'Time'),
            yaxis = dict(ticks = '', title = 'Frequency'),

        )

        fig = go.Figure(data = data, layout = layout)

        plot(fig, filename = self.output_dir + name + '.html')



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

            ys.append((y0, y1))
            xs.append((x0, x1))

        print 'There were', len(angles), 'detections'

        return xs, ys

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
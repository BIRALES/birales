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

    def __init__(self):
        return

    def remove_noise(self, noisy_data):
        length = len(noisy_data)
        corr = signal.correlate(noisy_data, np.ones(length), mode = 'same') / length

        return corr

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

    def visualise_line(self, name, intensity, time, frequency, xs, ys):
        traces = []
        for x, y in zip(xs, ys):
            traces.append(go.Scatter(
                x = x,
                y = y,
                name = 'Line',
                connectgaps = True
            ))

        traces.append(
            go.Heatmap(
                x = time,
                y = frequency,
                z = intensity,
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

    def line_detection(self, power, time, freq):
        filtered_power = power
        # for i, p in enumerate(power):
        #     filtered_power.append(self.remove_noise(p))


        image = np.array(filtered_power)
        # filtered_power = median(image, disk(1))

        # open_square = ndimage.binary_opening(image)
        #
        # eroded_square = ndimage.binary_erosion(image)
        # reconstruction = ndimage.binary_propagation(eroded_square, mask=image)
        # square = image
        # plt.figure(figsize=(9.5, 3))
        # plt.subplot(131)
        # plt.imshow(square, cmap=plt.cm.gray, interpolation='nearest')
        # plt.axis('off')
        # plt.subplot(132)
        # plt.imshow(open_square, cmap=plt.cm.gray, interpolation='nearest')
        # plt.axis('off')
        # plt.subplot(133)
        # plt.imshow(reconstruction, cmap=plt.cm.gray, interpolation='nearest')
        # plt.axis('off')
        #
        # plt.subplots_adjust(wspace=0, hspace=0.02, top=0.99, bottom=0.01, left=0.01, right=0.99)
        # plt.show()



        h, theta, d = hough_line(image)

        hough_transform_data = np.log(1 + h)
        # self.visualise_hough_transform('Hough Tansform', hough_transform_data)

        x0 = 0
        x1 = 600
        x = []
        y = []

        hspace, angles, dists = hough_line_peaks(h, theta, d)

        for _, angle, dist in zip(hspace, angles, dists):
            y0 = (dist - x0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)

            x.append([x0, x1])
            y.append([y0, y1])
            print x0, y0, x1, y1

        # self.visualise_line('Line', power, time, freq, x, y)
        self.visualise_line('Line', filtered_power, time, freq, x, y)

        return  # data = np.zeros(100)


od = OrbitDeterminationInputGenerator()
bd = BeamData()
power, time, freq = bd.mock_up()
# bd.visualise(power, time, freq, 'Mock_BeamData')

hough = od.line_detection(power, time, freq)

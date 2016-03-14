from plotly.offline import plot
import plotly.graph_objs as go
import numpy as np
from helpers import LineGeneratorHelper


class BeamData:
    noise_lvl = 0.2  # The standard deviation of the normal distribution noise
    output_dir = 'output/'

    def __init__(self):
        self.frequencies = np.linspace(0, 200, 200)
        self.time = 600
        self.power = self.mock_up()  # use mock up data for now

    def read_data(self, file_name):
        return

    def visualise(self, name):
        """
        Visualise the Beam Data as a Heatmap

        :param name: Name of the file
        :return: Void
        """
        data = [
            go.Heatmap(
                z = self.power,
                x = self.time,
                y = self.frequencies,
                colorscale = 'Viridis',
                colorbar = {'title': 'SNR'}, )
        ]

        layout = go.Layout(
            title = 'Beam Data',
            xaxis = dict(ticks = '', nticks = 36, title = 'Time'),
            yaxis = dict(ticks = '', title = 'Frequency'),

        )

        fig = go.Figure(data = data, layout = layout)

        plot(fig, filename = self.output_dir + name + '.html')

    def add_noise(self, noiseless_data):
        """
        Add noise to the passed numpy array
        :param noiseless_data:
        :return: noisy data
        """
        noise = abs(np.random.normal(0, self.noise_lvl, len(noiseless_data)))
        # noise = np.random.random(noiseless_data.shape) > 0.99

        return noiseless_data + noise

    def mock_up(self, visualise = False):
        """
        Build sample / mock up data to be used for testing
        :param visualise: Plot the mock data
        :return: power
        """
        power = np.zeros((max(self.frequencies), self.time))
        line_coor = LineGeneratorHelper.get_line((90, 120), (400, 180))  # (x0, y0), (x1, y1)

        for coordinate in line_coor:
            x = coordinate[0]
            y = coordinate[1]
            power[y][x] = 1.0

        if visualise:
            self.visualise('Mockup of Beam Data')

        print 'Detection has', len(line_coor),

        return power

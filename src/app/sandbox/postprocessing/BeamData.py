from plotly.offline import plot
from plotly.tools import FigureFactory
import plotly.graph_objs as go
import numpy as np


class BeamData:
    noise_lvl = 0.2  # The standard deviation of the normal distribution noise
    output_dir = 'output/'

    def __init__(self):
        self.frequencies = np.linspace(0, 200, 200)
        self.time = 600
        self.power = []

        self.mock_up()  # use mock up data for now

    def visualise(self, intensity, time, frequency, name):
        data = [
            go.Heatmap(
                z = intensity,
                x = time,
                y = frequency,
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

    def mock_up(self):
        count = 0

        power = []
        for f in self.frequencies:
            count += 1
            intensity = np.zeros(self.time)
            intensity = self.add_noise(intensity)

            if 150 < f < 180:
                intensity[count] = 1.0

            power.append(intensity)

        self.power = power

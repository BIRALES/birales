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

    def visualise(self, name):
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

    def mock_up2(self):
        count = 0

        power = []
        for f in self.frequencies:
            count += 1
            intensity = np.zeros(self.time)
            # intensity = self.add_noise(intensity)

            if 150 < f < 180:
                if count % 5 == 0:
                    intensity[count] = 1.0

            power.append(intensity)

        self.power = power

        self.visualise('Mockup of Beam Data')

    def mock_up(self):
        count = 0

        power = np.zeros((max(self.frequencies), self.time))
        line_coor = BeamData.get_line((90, 120), (400, 180))  # (x0, y0), (x1, y1)

        for coordinate in line_coor:
            x = coordinate[0]
            y = coordinate[1]
            # print x, y
            power[y][x] = 1.0
        # self.visualise('Mockup of Beam Data')

        self.power = power
        # self.visualise('Mockup of Beam Data')

        print 'Detection has', len(line_coor), 'pixels'

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

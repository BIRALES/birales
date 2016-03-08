import ephem
import numpy as np
import csv
from plotly.offline import plot
from plotly.tools import FigureFactory
import plotly.graph_objs as go


class DateTimeHelper:
    def __init__(self):
        return

    @staticmethod
    def juldate2ephem(date):
        """Convert Julian date to ephem date, measured from noon, Dec. 31, 1899."""
        return ephem.date(date - 2415020.)

    @staticmethod
    def ephem2juldate(date):
        """Convert ephem date (measured from noon, Dec. 31, 1899) to Julian date."""
        return float(date + 2415020.)


class TableMakerHelper:
    output_dir = 'output/'

    def __init__(self):
        self.headers = [
            'Epoch',
            'MJD2000',
            'Time Delay',
            'Doppler',
            'SNR'
        ]

        self.rows = []

    def create(self, headers, rows):
        self.set_headers(headers)
        self.set_rows(rows)

    def set_headers(self, headers):
        self.headers = headers

    def set_rows(self, rows):
        self.rows = rows

    def create_table(self):
        data_matrix = [self.headers]
        for row in self.rows:
            data_matrix.append(row)
        table = FigureFactory.create_table(data_matrix)

        return table

    def visualise(self, name):
        table = self.create_table()
        plot(table, filename = self.output_dir + name)

    def visualise_with_graph(self, name):
        table = self.create_table()
        time = []
        snr = []
        for row in self.rows:
            time.append(row[1])
            snr.append(row[4])

        # Make traces for graph
        trace1 = go.Scatter(x = time, y = snr,
                            marker = dict(color = '#0099ff'),
                            name = 'SNR v. Time',
                            xaxis = 'x2', yaxis = 'y2')

        table['data'].extend(go.Data([trace1]))

        # Edit layout for subplots
        table.layout.xaxis.update({'domain': [0, .65]})
        table.layout.xaxis2.update({'domain': [0.7, 1.]})

        # The graph's yaxis MUST BE anchored to the graph's xaxis
        table.layout.yaxis2.update({'anchor': 'x2'})
        table.layout.yaxis2.update({'title': 'SNR'})

        # Fix ranges
        table.layout.yaxis2.update({'range': [0, 8]})
        table.layout.xaxis2.update({'range': [min(time), max(time)]})
        table.layout.xaxis2.update({'title': 'Time'})

        # Update the margins to add a title and see graph x-labels.
        table.layout.margin.update({'t': 50, 'b': 120})
        table.layout.update({'title': 'SNR v. Time'})

        plot(table, filename = self.output_dir + name + '.html')


class BeamDataMockUp:
    def generate(self):
        go.Heatmap(
            z = z,
            x = date_list,
            y = programmers,
            colorscale = 'Viridis',
        )

    def create_heatmap(self):


class OrbitDeterminationInputGenerator:
    noise_lvl = 0.2  # The standard deviation of the normal distribution noise
    mock_data_dir = 'data/'

    def __init__(self):
        return

    def add_noise(self, noiseless_data):
        """
        Add noise to the passed numpy array
        :param noiseless_data:
        :return: noisy data
        """
        noise = abs(np.random.normal(0, self.noise_lvl, len(data)))
        return noiseless_data + noise

    def mock_data(self, file_name):
        with open(self.mock_data_dir + file_name) as csv_file:
            row = csv.reader(csv_file)
            mock_data = list(row)

        return mock_data

    def output(self):
        # get data
        # create output table
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


data = np.zeros(100)
od = OrbitDeterminationInputGenerator()
od.output()

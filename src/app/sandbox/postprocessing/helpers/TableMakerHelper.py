from plotly.offline import plot
from plotly.tools import FigureFactory
import plotly.graph_objs as go
import numpy as np
from app.sandbox.postprocessing.vendor import markup


class TableMakerHelper:

    def __init__(self):
        self.headers = [
            'Epoch',
            'MJD2000',
            'Time Delay',
            'Doppler',
            'SNR'
        ]

        self.rows = {}

        self.caption = None

    def create(self, headers, rows):
        self.set_headers(headers)
        self.set_rows(rows)

    def set_caption(self, caption):
        self.caption = caption

    def set_headers(self, headers):
        self.headers = headers

    def set_rows(self, rows):
        self.rows = rows

    def create_table(self):
        data_matrix = [self.headers]
        for i in range(0, 100):
            row = []
            for key in self.rows.keys():
                value = np.round(self.rows[key][i], 3)
                row.append(value)
            data_matrix.append(row)
        table = FigureFactory.create_table(data_matrix)

        return table

    def build_html_table(self, name):
        data_matrix = []
        for i in range(0, len(self.rows[self.rows.keys()[0]])):
            row = []
            for key in self.rows.keys():
                value = np.round(self.rows[key][i], 3)
                row.append(value)
            data_matrix.append(row)

        page = markup.page()
        page.init(title = name,
                  header = "Beam: 1",
                  css = 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css',
                  )

        page.table(class_ = 'table')

        page.caption(self.caption)

        page.tr()
        page.th(self.rows.keys())
        page.tr.close()
        for row in data_matrix:
            page.tr()
            page.td(row)
            page.tr.close()
        page.table.close()

        return str(page)

    def visualise(self, name):
        self.build_table(name)
        # table = self.create_table()
        # plot(table, filename = self.output_dir + name, auto_open = False)

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

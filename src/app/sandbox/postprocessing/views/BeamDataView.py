from plotly.offline import plot
import plotly.graph_objs as go
import os
import app.sandbox.postprocessing.config.application as config
import itertools


class BeamDataView:
    def __init__(self, name = 'beam'):
        self.layout = []
        self.data = []
        self.name = name
        self.shapes = []

    def set_layout(self, figure_title = 'Beam Data', x_axis_title = 'X Axis', y_axis_title = 'Y Axis'):
        """
        Set the layout configuration for the plotly figure

        :param figure_title: The title of the figure
        :param x_axis_title: The title of the x axis
        :param y_axis_title: The title of the y axis
        :return:
        """

        self.layout = go.Layout(
                title = figure_title,
                xaxis = dict(ticks = '', nticks = 36, title = x_axis_title),
                yaxis = dict(ticks = '', title = y_axis_title),
                shapes = self.shapes,
                # legend = dict(
                #         x = 0,
                #         y = 1,
                # )
        )

    def set_shapes(self, shapes):
        self.shapes = shapes

    def set_detections(self, detections):
        random_colors = itertools.cycle(['blue', 'pink', 'yellow', 'green', 'cyan', 'red'])
        for i, detection in enumerate(detections):
            self.data.append(go.Scatter(
                    x = detection[:, 0],
                    y = detection[:, 1],
                    mode = 'markers',
                    marker = dict(
                            color = next(random_colors),
                            # size = detection[:, 2]*5.  # size of particles is proportional to SNR
                    )))

    def set_data(self, data, x_axis, y_axis):
        """
        Set the data for the visualisation of the Beam data

        :param data: The data in the z axis for the heat map
        :param x_axis: The scale data in the x-axis
        :param y_axis: The scale data in the y-xis
        :return:
        """

        self.data.append(
                go.Heatmap(
                        z = data,
                        x = x_axis,
                        y = y_axis,
                        colorscale = 'Viridis',
                        colorbar = {'title': 'SNR'}, )
        )

    def save(self, file_path):
        """
        Visualise the Beam Data as a Heatmap using plot.ly

        :param file_path: Where to save the view file
        :return: Void
        """

        fig = go.Figure(data = self.data, layout = self.layout)
        parent = os.path.abspath(os.path.join(file_path + '.html', os.pardir))
        if not os.path.exists(parent):
            os.makedirs(parent)

        if config.VIEW_OUTPUT_FORMAT is 'HTML':
            ext = '.html'
        else:
            ext = '.png'

        plot(fig, filename = file_path + ext, auto_open = False)

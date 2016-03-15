from plotly.offline import plot
import plotly.graph_objs as go


class BeamDataVisualisation:
    def __init__(self, name):
        self.layout = []
        self.data = []
        self.name = name
        self.output_dir = 'output/'
        self.file_name = self.output_dir + name + '.html'

    def set_layout(self, figure_title = 'Beam Data', x_axis_title = 'X Axis', y_axis_title = 'Y Axis'):
        """
        Set the layout configruation for the plotly figure

        :param figure_title: The title of the figure
        :param x_axis_title: The title of the x axis
        :param y_axis_title: The title of the y axis
        :return:
        """
        self.layout = go.Layout(
            title = figure_title,
            xaxis = dict(ticks = '', nticks = 36, title = x_axis_title),
            yaxis = dict(ticks = '', title = y_axis_title),

        )

    def set_data(self, data, x_axis, y_axis):
        """
        Set the data for the visualisation of the Beam data

        :param data: The data in the z axis for the heat map
        :param x_axis: The scale data in the x-axis
        :param y_axis: The scale data in the y-xis
        :return:
        """
        self.data = [
            go.Heatmap(
                z = data,
                x = x_axis,
                y = y_axis,
                colorscale = 'Viridis',
                colorbar = {'title': 'SNR'}, )
        ]

    def show(self):
        """
        Visualise the Beam Data as a Heatmap using plot.ly

        :return: Void
        """

        fig = go.Figure(data = self.data, layout = self.layout)

        plot(fig, filename = self.file_name)

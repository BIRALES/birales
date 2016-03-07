from plotly.offline import plot
from plotly.graph_objs import Scatter
from plotly.graph_objs import Figure
from plotly.graph_objs import Layout
from plotly import tools

import numpy as np


class PostProcessing(object):
    post_directory = 'post/'

    @staticmethod
    def plot_wave(x, y):

        trace2 = Scatter(
                x=x,
                y=y,
                mode='circles',
                name='lines'
        )

        data = [trace2]

        plot(data, filename=PostProcessing.post_directory + '/scatter-mode.html')

    @staticmethod
    def plot_beam_pattern(bp, beam_former_id):
        data = [
            Scatter(
                    t=np.rad2deg(bp[0]),
                    r=bp[2],
                    mode='lines',
                    name='Beam',

                    marker=dict(
                            color='none',
                            line=dict(
                                    color='green',
                                    width=2,
                            )
                    )
            )
        ]

        layout = Layout(
                title='Beam Pattern',
                radialaxis={
                    'range': [-50, 0]
                },
                orientation=-90
        )

        fig = Figure(data=data, layout=layout)
        plot(fig, filename=PostProcessing.post_directory + '/beam_pattern' + beam_former_id + '.html')

    @staticmethod
    def plot_input_data(input_signals, time):
        data = []
        for i, input_signal in enumerate(input_signals):
            input_signal_trace = Scatter(
                    x=time,
                    y=input_signal,
                    name='Antenna Signal ' + str(i)
            )
            data.append(input_signal_trace)
        return data

    @staticmethod
    def plot_bf_data(bf_signals, time):
        data = []
        for i, bf in enumerate(bf_signals):
            antenna_signal_trace = Scatter(
                    x=time,
                    y=bf,
                    name='Delayed Antenna Signal ' + str(i)
            )
            data.append(antenna_signal_trace)

        return data

    @staticmethod
    def plot_results(beam_former, bf_signals, beam_former_id):
        input_signals = beam_former.INPUT_SIGNALS
        time = beam_former.TIME
        summation = beam_former.beamform(bf_signals)
        fig = tools.make_subplots(rows=1, cols=3,
                                  shared_yaxes=True,
                                  subplot_titles=('Input Signals', 'Delayed', 'Summation'))
        input_signals = PostProcessing.plot_input_data(input_signals, time)
        bf_data = PostProcessing.plot_bf_data(bf_signals, time)

        for input_signal in input_signals:
            fig.append_trace(input_signal, 1, 1)

        for bf in bf_data:
            fig.append_trace(bf, 1, 2)

        summation_trace = Scatter(
                x=time,
                y=summation,
                name='Beamformed Signal'
        )

        fig.append_trace(summation_trace, 1, 3)
        plot(fig, filename=PostProcessing.post_directory + '/beamformed_results' + beam_former_id + '.html')

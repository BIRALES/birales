import matplotlib.pyplot as plt
import numpy as np
from plotters import MultiLineMatplotlibPlotter, MultiWaterfallMatplotlibPlotter


class MultiBeamVisualisation:
    def __init__(self, beams, name):
        self.beams = beams
        self.name = name

    def waterfall(self):
        data = []
        for beam in self.beams:
            data.append(beam.snr.T)

        plotter = MultiWaterfallMatplotlibPlotter(fig_size=(16, 10),
                                                  fig_title='Waterfall',
                                                  plot_title='beam',
                                                  x_limits=(0, 8000),
                                                  y_limits=(0, 600),
                                                  x_label='Channel',
                                                  y_label='Time Sample',
                                                  data=data)
        plotter.save(self.name + '_waterfall')

    def bandpass(self):
        data = []
        for beam in self.beams:
            data.append(np.sum(beam.snr, axis=1))

        plotter = MultiLineMatplotlibPlotter(fig_size=(16, 10),
                                             fig_title='Bandpass',
                                             plot_title='beam',
                                             x_limits=(0, 8000),
                                             y_limits=(0, 3000),
                                             x_label='Channel',
                                             y_label='Power',
                                             data=data)
        plotter.save(self.name + '_bandpass')


class BeamVisualisation:
    def __init__(self, beam):
        self.beam = beam

    def visualise(self):
        """
        Naive plotting of the raw data. This is usually used to visualise quickly the beam data read
        :return: void
        """

        fig = plt.figure(figsize=(8, 8))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Beam %s" % self.beam.id)

        ax.imshow(self.beam.snr.transpose(),
                  aspect='auto',
                  origin='lower',
                  extent=[self.beam.channels[0],
                          self.beam.channels[-1], 0,
                          self.beam.time[-1]])

        ax.set_xlabel("Channel (kHz)")
        ax.set_ylabel("Time (s)")

        plt.show()

import os

import matplotlib.pyplot as plt
import numpy as np
from visualization.api.common.plotters import MultiLineMatplotlibPlotter, MultiWaterfallMatplotlibPlotter

# from pybirales.configuration.application import config


class MultiBeamVisualisation:
    def __init__(self, observation, data_set, name):
        self.name = name
        self.ext = '.png'
        self.output_dir = os.path.join(config.ROOT, config.get('io', 'MONITORING_FILE_PATH'), observation, data_set)
        self.water_fall_file_path = os.path.join(self.output_dir, self.name + '_waterfall' + self.ext)
        self.bandpass_file_path = os.path.join(self.output_dir, self.name + '_bandpass' + self.ext)

    def get_plot_file_path(self, plot_type):
        if plot_type == 'water_fall':
            return os.path.isfile(self.water_fall_file_path)
        if plot_type == 'bandpass':
            return os.path.isfile(self.bandpass_file_path)

        return False

    def waterfall(self, beam_data):
        data = []
        for beam in beam_data:
            data.append(beam.snr.T)

        plotter = MultiWaterfallMatplotlibPlotter(fig_size=(16, 10),
                                                  fig_title='Waterfall',
                                                  plot_title='beam',
                                                  x_limits='auto',
                                                  y_limits='auto',
                                                  x_label='Channel',
                                                  y_label='Time Sample',
                                                  data=beam_data)
        file_path = plotter.save(self.water_fall_file_path)

        return file_path

    def bandpass(self, beam_data):
        data = []
        for beam in beam_data:
            data.append(np.sum(beam.snr, axis=1))

        plotter = MultiLineMatplotlibPlotter(fig_size=(16, 10),
                                             fig_title='Bandpass',
                                             plot_title='beam',
                                             x_limits=(0, 8000),
                                             y_limits=(0, 3000),
                                             x_label='Channel',
                                             y_label='Power',
                                             data=data)
        plotter.save(self.bandpass_file_path)


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

from matplotlib import pyplot as plt
import h5py
import numpy as np


class FringeImager:

    def __init__(self, vis_original, vis_calib, no_of_antennas):

        self.vis_original = vis_original
        self.vis_calib = vis_calib
        self.no_of_antennas = no_of_antennas

    def plotter(self):

        with h5py.File(self.vis_original, "r") as f:
            data = f["Vis"]
            data_original = data[:]

        with h5py.File(self.vis_calib, "r") as f:
            data = f["Vis"]
            data_calib = data[:]

        counter = 0
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        f, axes = plt.subplots(2, sharex='all')
        for i in range(self.no_of_antennas):
            for j in range(i + 1, self.no_of_antennas):
                axes[0].plot(data_original[:, 0, counter, 0].real, linewidth=0.3)
                axes[0].set_title('Visibilities Before (Top) and After (Bottom) Calibration')
                axes[1].plot(data_calib[:, 0, counter, 0].real, linewidth=0.3)
                axes[1].set_xlabel('Time (sample no. from observation start)', size=9)
                axes[0].set_ylabel('Amplitude', size=9)
                axes[1].set_ylabel('Amplitude', size=9)
                axes[0].set_ylim([-1.0, 1.0])
                axes[1].set_ylim([-1.0, 1.0])

                counter += 1

        plt.show()
        # plt.savefig(self.calib_check_path, dpi=600)

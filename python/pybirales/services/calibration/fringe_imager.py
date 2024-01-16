import os

import h5py
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


class FringeImager:

    def __init__(self, vis_original, vis_calib, no_of_antennas):

        self.vis_original = vis_original
        self.vis_calib = vis_calib
        self.no_of_antennas = no_of_antennas

        self.calibration_check_path = os.path.join(os.path.dirname(vis_calib), 'fringes_check.png')

    def plotter(self):

        with h5py.File(self.vis_original, "r") as f:
            data = f["Vis"]
            data_original = data[:]

        with h5py.File(self.vis_calib, "r") as f:
            data = f["Vis"]
            data_calib = data[:]

        counter = 0

        # plt.rcParams["figure.figsize"] = (12, 7.416408)
        f, (ax1, ax2) = plt.subplots(2, 1, sharex='col')

        for i in range(self.no_of_antennas):
            for j in range(i + 1, self.no_of_antennas):
                ax1.plot(data_original[:, 0, counter, 0].real, linewidth=0.3)
                ax2.plot(data_calib[:, 0, counter, 0].real, linewidth=0.3)
                counter += 1

        ax1.set_title('Uncalibrated')
        ax1.set_ylabel('Amplitude')
        ax2.set_title('Calibrated')
        ax2.set_xlabel('Time sample')
        ax2.set_ylabel('Amplitude')

        # plt.tight_layout()
        # Save the calibration check image to disk
        plt.savefig(self.calibration_check_path)

        plt.close()

        return self.calibration_check_path

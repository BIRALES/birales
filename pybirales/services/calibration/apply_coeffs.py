import numpy as np
import h5py
import logging
# try:
#     from casacore import tables
# except ImportError:
#     logging.warning('casacore not found.')

from pybirales.services.calibration import fringe_imager


class CoeffManager:

    def __init__(self):

        self.coeffs = None
        self.baseline_no = None
        self.coeff_filepath = None
        self.calib_filepath = None
        self._logger = logging.getLogger(__name__)

    def read_coeffs(self, coeffs_in):

        self.coeffs = coeffs_in

    def save_coeffs(self, coeffs_filepath):

        self.coeff_filepath = coeffs_filepath
        text_file = open(self.coeff_filepath, "w")
        for pol in range(self.coeffs.shape[1]):
            for i in range(self.coeffs.shape[0]):
                text_file.write(format(self.coeffs[i, pol], ".16e") + '\n')
        text_file.close()

    def coeffs_apply(self, real_vis, baseline_no, main_dir, i):

        # Create empty calibrated visibilities matrix
        vis_calib = np.ones((len(real_vis[:, 0]), len(real_vis[0, :])), dtype=np.complex)

        for pol in range(real_vis.shape[1]):
            coeffs_real = []
            coeffs_imag = []
            for j in range(self.coeffs.shape[0]):
                coeffs_real.append(self.coeffs[j, pol].real)
                coeffs_imag.append(self.coeffs[j, pol].imag)

            coeffs_real = np.array(coeffs_real)
            coeffs_imag = np.array(coeffs_imag)

            # Calibrate visibilities
            for t in range(len(real_vis[:, 0])):
                a1 = int(baseline_no[t, 0])
                a2 = int(baseline_no[t, 1])
                vis_calib[t, pol] = real_vis[t, pol] * ((np.complex(coeffs_real[a1], coeffs_imag[a1])) *
                                                        np.conj(np.complex(coeffs_real[a2], coeffs_imag[a2])))

        self.calib_filepath = main_dir + '/calib' + str(i) + '.h5'
        # Write calibrated visibilities to h5 file
        f2 = h5py.File(self.calib_filepath, 'w')
        name = 'Vis'
        dset = f2.create_dataset(name, (len(real_vis[:, 0]), len(real_vis[0, :])), dtype='c16')
        dset[:, :] = vis_calib[:, :]
        f2.flush()
        f2.close()

    def coeffs_apply_transit(self, obs_file, baseline_no, main_dir, calib_check_path, i):

        with h5py.File(obs_file, "r") as f:
            data = f["Vis"]
            real_vis = data[:, 0, :, 0]

        # Create empty calibrated visibilities matrix
        vis_calib = np.ones((len(real_vis[:, 0]), len(real_vis[0, :])), dtype=np.complex)

        for basl in range(real_vis.shape[1]):
            coeffs_real = []
            coeffs_imag = []
            for j in range(self.coeffs.shape[0]):
                coeffs_real.append(self.coeffs[j, 0].real)
                coeffs_imag.append(self.coeffs[j, 0].imag)

            coeffs_real = np.array(coeffs_real)
            coeffs_imag = np.array(coeffs_imag)

            # Calibrate visibilities
            for t in range(real_vis.shape[0]):
                a1 = int(baseline_no[basl, 0])
                a2 = int(baseline_no[basl, 1])
                vis_calib[t, basl] = real_vis[t, basl] * ((np.complex(coeffs_real[a1], coeffs_imag[a1])) *
                                                          np.conj(np.complex(coeffs_real[a2], coeffs_imag[a2])))

        self.calib_filepath = calib_check_path

        # Write calibrated visibilities to h5 file
        f2 = h5py.File(self.calib_filepath, 'w')
        name = 'Vis'
        dset = f2.create_dataset(name, (len(real_vis[:, 0]), 1, len(real_vis[0, :]), 1), dtype='c16')
        vis_calib = np.reshape(vis_calib, [len(real_vis[:, 0]), 1, len(real_vis[0, :]), 1])
        dset[:, :, :, :] = vis_calib[:, :, :, :]
        f2.flush()
        f2.close()

    def fringe_imager(self, transit_file, calib_check_path, no_of_antennas):

        fringes = fringe_imager.FringeImager(transit_file, calib_check_path, no_of_antennas)
        self._logger.info('Plotting fringes before and after calibration')

        return fringes.plotter()


class CoeffManagerRun:

    def __init__(self, cal_input, vis_in, coeffs_in):
        self.coeffs_manager = CoeffManager()
        self.__baseline_no = None
        self.__uncalib_vis = vis_in

        self.__coeffs_filepath = cal_input['coeffs_filepath']
        self.__model_generation = cal_input['model_generation']
        self.__test_run_check = cal_input['test_run']
        self.__transit_run = cal_input['transit_run']
        self.__obs_file = cal_input['obs_file']
        self.__transit_file = cal_input['transit_file']
        self.__main_dir = cal_input['main_dir']
        self.__no_of_antennas = cal_input['no_of_antennas']
        self.__calib_check_path = cal_input['calib_check_path']
        self.__ms_dir = cal_input['model_vis_directory']
        if self.__test_run_check is False:
            self.__baseline_no = cal_input['baseline_no']

        self.coeffs_manager.read_coeffs(coeffs_in)
        self.coeffs_manager.save_coeffs(self.__coeffs_filepath)

        self.__vis_in = vis_in

    def check(self):
        if not self.__test_run_check:
            if not self.__transit_run:
                self.coeffs_manager.coeffs_apply(self.__vis_in, self.__baseline_no, self.__main_dir, 0)

            if self.__transit_run is True:
                self.coeffs_manager.coeffs_apply_transit(self.__obs_file, self.__baseline_no, self.__main_dir,
                                                         self.__calib_check_path, 0)

                return self.coeffs_manager.fringe_imager(self.__transit_file, self.__calib_check_path, self.__no_of_antennas)

        return None
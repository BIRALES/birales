import logging
from copy import deepcopy

import numpy as np
import scipy.linalg as lin

from pybirales.services.calibration import geometric as gu


class SelfCal(object):
    """
    The class runs a calibration routine, using StEFCal, assuming a point source at main beam centre, in order to
    retrieve per element calibration coefficients.

    """

    def __init__(self):

        """
        Default Initializer

        """

        self.latest_coeffs = None
        self.coeffs_no_geom = None
        self.coeffs_out = None
        self.coeff_type = None
        self._logger = logging.getLogger(__name__)

    def stefcal_run(self, vis_in, no_of_antennas):

        max_iterations = 800
        tolerance = 1e-20

        """
        Calibration with StEFCal
        """

        self.latest_coeffs = np.ones((no_of_antennas, vis_in.shape[1]), dtype=np.complex64)

        counter = 0
        measured = np.zeros((no_of_antennas, no_of_antennas), dtype=np.complex64)
        for i in range(no_of_antennas):
            for j in range(i + 1, no_of_antennas):
                measured[i, j] = vis_in[counter]
                measured[j, i] = np.conj(vis_in[counter])
                counter += 1

        # Define model as a unity-valued corr matrix with a zeroed diag
        model = np.ones_like(measured, dtype=np.complex64)

        for i in range(no_of_antennas):
            model[i, i] = 0. + 0.j

        # StEFCal run
        nst = len(model)
        numsamples = measured.shape[0] / measured.shape[1]

        gs = []
        gs = gs + [np.ones(nst, dtype=np.complex64)]

        model = np.tile(model, (numsamples, 1))

        norms = []

        for i in range(1, max_iterations):

            g = np.ones(nst, dtype=np.complex64)

            for p in range(0, nst):

                Zp = np.multiply(gs[i - 1], model[:, p])
                if np.vdot(Zp, Zp) != 0.:
                    g[p] = np.divide(np.vdot(measured[:, p], Zp), np.vdot(Zp, Zp))
                else:
                    g[p] = 0. + 0.j

            gs = gs + [g]

            if np.mod(i, 2) == 0 and i > 0:

                norm = lin.norm(gs[i] - gs[i - 1]) / lin.norm(gs[i])

                norms = norms + [norm]

                if norm >= tolerance:

                    gs[i] = (gs[i] + gs[i - 1]) / 2

                else:
                    break

        inverter = np.ones(len(gs[i - 1]), dtype=np.complex64)
        gs_out = np.divide(inverter, gs[i - 1], out=np.zeros_like(inverter), where=gs[i - 1] != 0.)

        self.latest_coeffs[:, 0] = gs_out

        self.coeff_type = "calib_geom"
        self._logger.info('StEFCal calibration successful')

    def geometric_removal(self, vis_in, no_of_antennas, calib_dec, antennas, longitude, latitude, frequency, bandwidth):

        config = dict()

        config['reference_antenna_location'] = [longitude, latitude]
        config['reference_declination'] = calib_dec
        config['pointings'] = [[0, 0]]
        config['antenna_locations'] = antennas

        config['nbeams'] = len(config['pointings'])
        config['start_center_frequency'] = frequency / 1e6
        config['channel_bandwidth'] = bandwidth / 1e6

        pointing = gu.Pointing(config, vis_in.shape[1], no_of_antennas)

        for pol in range(self.latest_coeffs.shape[1]):
            self.latest_coeffs[:, pol] *= np.conj(pointing.weights[0, 0, :])

        self.coeffs_no_geom = deepcopy(self.latest_coeffs)
        self.coeff_type = "no_geom"
        self._logger.info('Geometric calibration pointing coefficient removal successful')

    def geometric_addition(self, vis_in, no_of_antennas, point_dec, antennas, longitude, latitude, frequency, bandwith):

        config = dict()

        config['reference_antenna_location'] = [longitude, latitude]
        config['reference_declination'] = point_dec
        config['pointings'] = [[0, 0]]
        config['antenna_locations'] = antennas

        config['nbeams'] = len(config['pointings'])
        config['start_center_frequency'] = frequency / 1e6
        config['channel_bandwidth'] = bandwith / 1e6

        pointing = gu.Pointing(config, vis_in.shape[1], no_of_antennas)

        for pol in range(self.latest_coeffs.shape[1]):
            self.latest_coeffs[:, pol] *= pointing.weights[0, 0, :]

        self.coeff_type = "pointing_geom"
        self._logger.info('Geometric observation pointing coefficient addition successful')

    def save_coeffs(self, main_dir):

        """
        To be used for saving coefficients to text file during testing.

        :param main_dir: Main directory for saving coefficients files

        """

        coeffs = self.latest_coeffs

        # save calibration coefficients
        if self.coeff_type == "calib_geom":
            self.coeffs_out = main_dir + '/coeffs_raw.txt'
        if self.coeff_type == "no_geom":
            self.coeffs_out = main_dir + '/coeffs_no_geom.txt'
        if self.coeff_type == "pointing_geom":
            self.coeffs_out = main_dir + '/coeffs_pointed.txt'

        text_file = open(self.coeffs_out, 'w')

        for pol in range(coeffs.shape[1]):

            for k in range(coeffs.shape[0]):
                if coeffs[k, pol].imag >= 0:
                    text_file.write('%.16e' % coeffs[k, pol].real + '+' + '%.16e' % coeffs[k, pol].imag + 'j' + '\n')
                if coeffs[k].imag < 0:
                    text_file.write('%.16e' % coeffs[k, pol].real + '%.16e' % coeffs[k, pol].imag + 'j' + '\n')

        text_file.close()


class SelfCalRun:

    def __init__(self, cal_input, vis_in):
        self.selfcal = SelfCal()
        self._vis_in = vis_in

        self._no_of_antennas = cal_input['no_of_antennas']
        self._pointing_ra = cal_input['pointing_ra']
        self._pointing_dec = cal_input['pointing_dec']
        self._calibration_dec = cal_input['calibration_dec']
        self._antennas = cal_input['antennas']
        self._longitude = cal_input['longitude']
        self._latitude = cal_input['latitude']
        self._frequency = cal_input['frequency']
        self._bandwidth = cal_input['bandwidth']
        self._main_dir = cal_input['main_dir']

        # reshape inbound visibilities if necessary
        if len(self._vis_in.shape) < 2:
            self._vis_in = np.reshape(self._vis_in, [self._vis_in.shape[0], 1])

        # calibrate using StEFCal
        self.selfcal.stefcal_run(self._vis_in, self._no_of_antennas)

        self.selfcal.save_coeffs(self._main_dir)

        # remove geometric coefficients for calibrator coordinates
        self.selfcal.geometric_removal(self._vis_in, self._no_of_antennas, self._calibration_dec,
                                       self._antennas, self._longitude, self._latitude, self._frequency,
                                       self._bandwidth)

        self.selfcal.save_coeffs(self._main_dir)

        # add geometric coefficients for observation coordinates
        self.selfcal.geometric_addition(self._vis_in, self._no_of_antennas, self._pointing_dec,
                                        self._antennas, self._longitude, self._latitude, self._frequency,
                                        self._bandwidth)

        self.selfcal.save_coeffs(self._main_dir)

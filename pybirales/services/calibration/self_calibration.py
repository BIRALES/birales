import numpy as np
import logging
from copy import deepcopy

import geometric as gu


class SelfCal(object):
    """
    The SelfCal class runs a simple self-calibration scheme assuming a point source at zenith, in order to obtain
    per antenna calibration coefficients.

    """

    def __init__(self):

        """
        Default Initializer

        """

        self.bas_coeffs = None
        self.ant_coeffs = None
        self.latest_coeffs = None
        self.coeffs_calib_geom = None
        self.coeffs_no_geom = None
        self.gain_coeffs = None
        self.phase_coeffs = None
        self.peak_visibilities = None
        self.coeffs_out = None
        self.vis_transfer = None
        self.coeff_type = None
        self.ref_antenna = None
        self._logger = logging.getLogger(__name__)

    def phasecal_run(self, vis_in, no_of_antennas, baseline_no):

        """
        Carries out a sky-independent phase calibration. Input visibilities can be either from an incoming stream
        of calibrator-source-beamformed data or as found peak visibilities from the test_vis_peak_find function.

        :param vis_in: A visibilities vector, containing antenna visibilities in an ordered format and containing only
        0.5(n^2-n) visibilities, starting from visibility between antennas 0 and 1, followed by that between antennas
        0 and 2, and so on until that between antennas n-1 and n
        :param no_of_antennas: The number of antennas, n
        :param baseline_no: The baseline number per antenna pair

        """

        self.phase_coeffs = np.ones((vis_in.shape[1], no_of_antennas), dtype=np.complex64)

        self.bas_coeffs = np.ones((vis_in.shape[0], vis_in.shape[1]), dtype=np.complex64)

        for pol in range(vis_in.shape[1]):

            with np.errstate(divide='ignore', invalid='ignore'):
                self.bas_coeffs = np.max(vis_in[:, pol]) / np.array(vis_in[:, pol])

            # Select visibilities for respective reference antenna
            self.ref_antenna = 0
            selection = np.where(baseline_no[:, :] == self.ref_antenna)[0]
            counter = 0
            for i in range(no_of_antennas):
                if i < self.ref_antenna:
                    self.phase_coeffs[pol, i] = self.bas_coeffs[selection[counter]]
                    counter += 1
                if i > self.ref_antenna:
                    self.phase_coeffs[pol, i] = np.conj(self.bas_coeffs[selection[counter]])
                    counter += 1

            coeffs_mean = np.mean(self.phase_coeffs[pol, :])
            coeffs_std = np.std(self.phase_coeffs[pol, :])
            for i in range(0, len(self.phase_coeffs[pol, :])):
                if (coeffs_mean - (2 * coeffs_std)) > self.phase_coeffs[pol, i]:
                    self.phase_coeffs[pol, i] = 1. + 0.j
                if (coeffs_mean + (2 * coeffs_std)) < self.phase_coeffs[pol, i]:
                    self.phase_coeffs[pol, i] = 1. + 0.j

        self.ant_coeffs = deepcopy(self.phase_coeffs)
        self.ant_coeffs = np.transpose(self.ant_coeffs)
        self.coeff_type = "calib_geom"
        self._logger.info('Phase calibration successful')

    def gaincal_run(self, vis_in, no_of_antennas):

        """
        Carries out a sky-independent gain calibration. Input visibilities can be either from an incoming stream
        of calibrator-source-beamformed data, as found peak visibilities from the test_vis_peak_find function or as
        updated with the

        :param vis_in: A visibilities vector, containing antenna visibilities in an ordered format and containing only
        0.5(n^2-n) visibilities, starting from visibility between antennas 0 and 1, followed by that between antennas
        0 and 2, and so on until that between antennas n-1 and n
        :param no_of_antennas: The number of antennas, n

        """

        power_in = np.zeros((vis_in.shape[0], vis_in.shape[1]))

        for pol in range(vis_in.shape[1]):
            # vis_in[:, pol] /= np.max(vis_in[:, pol])
            for i in range(vis_in.shape[0]):
                power_in[i, pol] = vis_in[i, pol].real  # ((vis_in[i, pol].real**2) + (vis_in[i, pol].imag**2))

        self.gain_coeffs = np.ones((vis_in.shape[1], no_of_antennas), dtype=np.complex)

        for pol in range(vis_in.shape[1]):

            a_mat = np.zeros((vis_in.shape[0], no_of_antennas))
            b_vec = np.zeros(vis_in.shape[0], dtype=np.complex64)

            counter = 0

            for i in range(0, (no_of_antennas - 1)):
                for j in range((i + 1), no_of_antennas):
                    a_mat[counter, i] = 1
                    a_mat[counter, j] = 1
                    with np.errstate(divide='ignore', invalid='ignore'):
                        b_vec[counter] = np.log(power_in[counter, pol])
                    counter += 1

            with np.errstate(divide='ignore', invalid='ignore'):
                log_coeff = np.linalg.lstsq(a_mat, b_vec)[0]
            self.gain_coeffs[pol, :] = 10 ** log_coeff

            # self.gain_coeffs[pol, :] /= self.gain_coeffs[pol, 0]

            coeffs_mean = np.mean(self.gain_coeffs[pol, :])
            coeffs_std = np.std(self.gain_coeffs[pol, :])
            for i in range(0, len(self.gain_coeffs[pol, :])):
                if (coeffs_mean - (2 * coeffs_std)) > self.gain_coeffs[pol, i]:
                    self.gain_coeffs[pol, i] = 1. + 0.j
                if (coeffs_mean + (2 * coeffs_std)) < self.gain_coeffs[pol, i]:
                    self.gain_coeffs[pol, i] = 1. + 0.j

        self.ant_coeffs = deepcopy(self.gain_coeffs)
        self.ant_coeffs = np.transpose(self.ant_coeffs)
        self.coeff_type = "calib_geom"
        self._logger.info('Gain calibration successful')

    def coeff_quick_apply(self, vis_in, baseline_no):

        # Create empty calibrated visibilities matrix
        self.vis_transfer = np.ones((len(vis_in[:, 0]), len(vis_in[0, :])), dtype=np.complex64)

        for pol in range(vis_in.shape[1]):
            coeffs_real = []
            coeffs_imag = []
            for j in range(self.ant_coeffs.shape[0]):
                coeffs_real.append(self.ant_coeffs[j, pol].real)
                coeffs_imag.append(self.ant_coeffs[j, pol].imag)

            coeffs_real = np.array(coeffs_real)
            coeffs_imag = np.array(coeffs_imag)

            # Calibrate visibilities
            for t in range(len(vis_in[:, 0])):
                a1 = int(baseline_no[t, 0])
                a2 = int(baseline_no[t, 1])
                self.vis_transfer[t, pol] = vis_in[t, pol] * ((np.complex(coeffs_real[a1], coeffs_imag[a1])) *
                                                              np.conj(np.complex(coeffs_real[a2], coeffs_imag[a2])))

    def coeff_apply(self, vis_in, baseline_no):

        # Create empty calibrated visibilities matrix
        self.vis_transfer = np.ones((len(vis_in[:, 0]), len(vis_in[0, :])), dtype=np.complex)

        for pol in range(vis_in.shape[1]):
            coeffs_real = []
            coeffs_imag = []
            for j in range(self.latest_coeffs.shape[1]):
                coeffs_real.append(self.latest_coeffs[pol, j].real)
                coeffs_imag.append(self.latest_coeffs[pol, j].imag)

            coeffs_real = np.array(coeffs_real)
            coeffs_imag = np.array(coeffs_imag)

            # Calibrate visibilities
            for t in range(len(vis_in[:, 0])):
                a1 = int(baseline_no[t, 0])
                a2 = int(baseline_no[t, 1])
                self.vis_transfer[t, pol] = vis_in[t, pol] * ((np.complex(coeffs_real[a1], coeffs_imag[a1])) *
                                                              np.conj(np.complex(coeffs_real[a2], coeffs_imag[a2])))

    def coeff_manager(self):

        for pol in range(self.ant_coeffs.shape[1]):
            for i in range(0, self.ant_coeffs.shape[0]):
                self.latest_coeffs[pol, i] *= self.ant_coeffs[i, pol]

        # Reset phase coefficients
        self.ant_coeffs = np.ones((self.ant_coeffs.shape[0], self.ant_coeffs.shape[1]), dtype=np.complex64)

    def coeff_builder(self):

        self.ant_coeffs = np.ones((self.phase_coeffs.shape[0], self.phase_coeffs.shape[1]), dtype=np.complex)

        for pol in range(self.phase_coeffs.shape[0]):
            for i in range(0, self.phase_coeffs.shape[1]):
                # real_calibration = self.phase_coeffs[pol, i].real * self.gain_coeffs[pol, i]
                # self.ant_coeffs[pol, i] = np.complex(real_calibration, self.phase_coeffs[pol, i].imag)
                self.ant_coeffs[pol, i] = self.latest_coeffs[i, pol] * self.gain_coeffs[pol, i]

                # self.ant_coeffs[pol, :] /= self.ant_coeffs[pol, 0]

            # self.ant_coeffs[pol, :].real = self.ant_coeffs[pol, :].real / self.ant_coeffs[pol, 0].real
            # self.ant_coeffs[pol, :].imag = self.ant_coeffs[pol, :].imag - self.ant_coeffs[pol, 0].imag

        self.ant_coeffs = np.transpose(self.ant_coeffs)
        self.latest_coeffs = deepcopy(self.ant_coeffs)
        self.coeff_type = "calib_geom"
        self.coeffs_calib_geom = self.ant_coeffs

    def phase_coeff_builder(self):

        self.ant_coeffs = np.ones((self.phase_coeffs.shape[0], self.phase_coeffs.shape[1]), dtype=np.complex)

        for pol in range(self.phase_coeffs.shape[0]):
            for i in range(0, self.phase_coeffs.shape[1]):
                # real_calibration = self.phase_coeffs[pol, i].real * self.gain_coeffs[pol, i]
                # self.ant_coeffs[pol, i] = np.complex(real_calibration, self.phase_coeffs[pol, i].imag)
                self.ant_coeffs[pol, i] = self.latest_coeffs[pol, i] * self.phase_coeffs[pol, i]

                # self.ant_coeffs[pol, :] /= self.ant_coeffs[pol, 0]

            # self.ant_coeffs[pol, :].real = self.ant_coeffs[pol, :].real / self.ant_coeffs[pol, 0].real
            # self.ant_coeffs[pol, :].imag = self.ant_coeffs[pol, :].imag - self.ant_coeffs[pol, 0].imag

        self.ant_coeffs = np.transpose(self.ant_coeffs)
        self.latest_coeffs = deepcopy(self.ant_coeffs)
        self.coeff_type = "calib_geom"
        self.coeffs_calib_geom = self.ant_coeffs

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
            self.latest_coeffs[:, pol] /= pointing.weights[0, 0, :]
            # self.ant_coeffs[:, pol].real /= pointing.weights[0, 0, :].real
            # self.ant_coeffs[:, pol].imag -= pointing.weights[0, 0, :].imag
            # self.ant_coeffs[:, pol].real /= self.ant_coeffs[0, pol].real
            # self.ant_coeffs[:, pol].imag -= self.ant_coeffs[0, pol].imag

            self.ref_antenna = 0
            self.latest_coeffs[:, pol] /= self.latest_coeffs[self.ref_antenna, pol]
        self.coeffs_no_geom = deepcopy(self.latest_coeffs)
        # self.coeffs_no_geom.imag = np.degrees(self.coeffs_no_geom.imag)
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
            # self.ant_coeffs[:, pol] = np.conj(self.ant_coeffs[:, pol])
            # self.ant_coeffs[:, pol].real *= pointing.weights[0, 0, :].real
            # self.ant_coeffs[:, pol].imag += pointing.weights[0, 0, :].imag
            # self.ant_coeffs[:, pol].real /= self.ant_coeffs[0, pol].real
            # self.ant_coeffs[:, pol].imag -= self.ant_coeffs[0, pol].imag

        self.coeff_type = "pointing_geom"
        self._logger.info('Geometric observation pointing coefficient addition successful')

    def transit_peak_find(self, observation_in):

        self.peak_visibilities = observation_in[:]

        # self._logger.info('Transit observation peak observed at t={}'.format(vis_peak))

    def test_save_coeffs(self, coeffs, main_dir):

        """
        To be used for saving coefficients to text file during testing.

        :param coeffs: Array containing antenna coefficients, in order from antenna 0 to antenna n
        :param main_dir: Main directory for saving coefficients files

        """

        # Save Calibration Coefficients
        if self.coeff_type == "calib_geom":
          self.coeffs_out = main_dir + '/coeffs_raw.txt'
        if self.coeff_type == "no_geom":
          self.coeffs_out = main_dir + '/coeffs_no_geom.txt'
        if self.coeff_type == "pointing_geom":
          self.coeffs_out = main_dir + '/coeffs_pointed.txt'
        text_file = open(self.coeffs_out, 'w')

        for pol in range(coeffs.shape[1]):

            coeffs_real = []
            coeffs_imag = []
            for i in range(len(coeffs)):
                coeffs_real.append(coeffs[i, pol].real)
                coeffs_imag.append(coeffs[i, pol].imag)

            coeffs_real = np.array(coeffs_real)
            coeffs_imag = np.array(coeffs_imag)

            coeffs_out = np.zeros(len(coeffs_real), dtype=np.complex)
            for i in range(len(coeffs_real)):
                coeffs_out[i] = np.complex(coeffs_real[i], coeffs_imag[i])

            for k in range(coeffs_out.shape[0]):
                if coeffs_out[k].imag >= 0:
                    text_file.write('%f' % coeffs_out[k].real + '+' + '%f' % coeffs_out[k].imag + 'j' + '\n')
                if coeffs_out[k].imag < 0:
                    text_file.write('%f' % coeffs_out[k].real + '%f' % coeffs_out[k].imag + 'j' + '\n')

        text_file.close()

        # if self.coeff_type == "no_geom":
        #     for pol in range(coeffs.shape[1]):
        #         self.coeffs_out = str(calib_dir) + '/' + obs_time + '_' + str(declination) + '.npy'
        #         np.save(self.coeffs_out, coeffs[:, pol])
        #         print('No Geom = ', coeffs[:, pol])

    def combined_run(self, vis_in, no_of_antennas):

        """
        Carries out a simple, sky-independent self-calibration. Input visibilities can be either from an incoming stream
        of calibrator-source-beamformed data or as found peak visibilities from the test_vis_peak_find function.

        :param vis_in: A visibilities vector, containing antenna visibilities in an ordered format and containing only
        0.5(n^2-n) visibilities, starting from visibility between antennas 0 and 1, followed by that between antennas
        0 and 2, and so on until that between antennas n-1 and n
        :param no_of_antennas: The number of antennas, n

        """

        self.phase_coeffs = np.ones((vis_in.shape[1], no_of_antennas), dtype=np.complex64)
        self.bas_coeffs = np.ones((vis_in.shape[0], vis_in.shape[1]), dtype=np.complex64)

        for pol in range(vis_in.shape[1]):

            with np.errstate(divide='ignore', invalid='ignore'):
                self.bas_coeffs[:, pol] = 1.0 / np.array(vis_in[:, pol])

            a_mat = np.zeros((vis_in.shape[0], no_of_antennas))
            b_vec = np.zeros(vis_in.shape[0], dtype=np.complex)

            counter = 0
            for i in range(0, (no_of_antennas - 1)):
                for j in range((i + 1), no_of_antennas):
                    a_mat[counter, i] = 1
                    a_mat[counter, j] = 1
                    with np.errstate(divide='ignore', invalid='ignore'):
                        b_vec[counter] = np.log(self.bas_coeffs[counter, pol])
                    counter += 1

            with np.errstate(divide='ignore', invalid='ignore'):
                log_coeff = np.linalg.lstsq(a_mat, b_vec)[0]
            self.phase_coeffs[pol, :] = 10. ** np.sqrt(log_coeff)

            coeffs_mean = np.mean(self.phase_coeffs[pol, :])
            coeffs_std = np.std(self.phase_coeffs[pol, :])
            for i in range(0, len(self.phase_coeffs[pol, :])):
                if (coeffs_mean - coeffs_std) > self.phase_coeffs[pol, i]:
                    self.phase_coeffs[pol, i] = 1. + 0.j
                if (coeffs_mean + coeffs_std) < self.phase_coeffs[pol, i]:
                    self.phase_coeffs[pol, i] = 1. + 0.j

        self.ant_coeffs = self.phase_coeffs
        self.ant_coeffs = np.transpose(self.ant_coeffs)


class SelfCalRun:

    def __init__(self, cal_input, vis_in):

        self.selfcal = SelfCal()
        self._vis_in = None
        self._original_vis_in = None

        self._transit_observation = False
        if cal_input['transit_run'] is True:
            self._transit_observation = True

        self._no_of_antennas = cal_input['no_of_antennas']
        self._gaincal = cal_input['gaincal']
        self._phasecal = cal_input['phasecal']
        self._stefcal = cal_input['stefcal']
        self._pointing_ra = cal_input['pointing_ra']
        self._pointing_dec = cal_input['pointing_dec']
        self._calibration_dec = cal_input['calibration_dec']
        self._antennas = cal_input['antennas']
        self._longitude = cal_input['longitude']
        self._latitude = cal_input['latitude']
        self._frequency = cal_input['frequency']
        self._bandwidth = cal_input['bandwidth']
        self._baseline_no = cal_input['baseline_no']
        self._main_dir = cal_input['main_dir']
        self._obs_time = cal_input['obs_time']
        self._cal_dir = cal_input['cal_coeffs_dir']

        self._vis_in = vis_in
        self.selfcal.vis_transfer = self._vis_in

        if self._transit_observation is True:
            self.selfcal.transit_peak_find(self._vis_in)
            self._vis_in = self.selfcal.peak_visibilities
            self.selfcal.vis_transfer = self._vis_in

        self.selfcal.latest_coeffs = np.ones((1, self._no_of_antennas), dtype=np.complex64)

        if len(self._vis_in.shape) < 2:
            self._vis_in = np.reshape(self.selfcal.peak_visibilities, [self.selfcal.peak_visibilities.shape[0], 1])
            self.selfcal.vis_transfer = self._vis_in

        if self._phasecal is True and self._gaincal is True:

            counter = 0
            while True:

                self.selfcal.phasecal_run(self.selfcal.vis_transfer, self._no_of_antennas, self._baseline_no)
                self.selfcal.coeff_manager()
                self.selfcal.coeff_apply(self._vis_in, self._baseline_no)

                if counter >= 0 or np.abs(np.abs(np.max(self.selfcal.vis_transfer).real) -
                                          np.abs(np.min(self.selfcal.vis_transfer).real)) < 0.1:
                    break

                #delta = np.abs(np.abs(np.max(self.selfcal.vis_transfer).real) -
                #               np.abs(np.min(self.selfcal.vis_transfer).real))

                #self.selfcal.gaincal_run(self.selfcal.vis_transfer, self._no_of_antennas)
                #self.selfcal.coeff_manager()
                #self.selfcal.coeff_apply(self._vis_in, self._baseline_no)

                #if delta < np.abs(np.abs(np.max(self.selfcal.vis_transfer).real) -
                #                  np.abs(np.min(self.selfcal.vis_transfer).real)) or \
                #        np.abs(np.abs(np.max(self.selfcal.vis_transfer).real) -
                #              np.abs(np.min(self.selfcal.vis_transfer).real)) < 0.1:
                #    break

                if counter >= 10:
                    break

                counter += 1

        self.selfcal.latest_coeffs = np.transpose(self.selfcal.latest_coeffs)

        self.selfcal.test_save_coeffs(self.selfcal.latest_coeffs, self._main_dir)
        self.selfcal.geometric_removal(self._vis_in, self._no_of_antennas, self._calibration_dec,
                                       self._antennas, self._longitude, self._latitude, self._frequency,
                                       self._bandwidth)
        #self.selfcal.coeffs_no_geom = self.selfcal.latest_coeffs
        self.selfcal.test_save_coeffs(self.selfcal.coeffs_no_geom, self._main_dir)
        self.selfcal.geometric_addition(self._vis_in, self._no_of_antennas, self._pointing_dec,
                                        self._antennas, self._longitude, self._latitude, self._frequency,
                                        self._bandwidth)
        self.selfcal.test_save_coeffs(self.selfcal.latest_coeffs, self._main_dir)

        # if self._stefcal is False:
        #     forward[0] = [self.selfcal.latest_coeffs, message.Default[0], message.Default[1], message.Default[2]]
        #
        # if self._stefcal is True:
        #     forward[0] = [message.Default[0], message.Default[1], message.Default[2], self.selfcal.latest_coeffs,
        #                   message.Default[3], message.Default[4]]

import logging
import struct
import time

import corr
import numpy as np

from pybirales.utilities.singleton import Singleton
from pybirales import settings


@Singleton
class Backend(object):
    def __init__(self):
        """ BiralesBackend constructor """
        # Connect to ROACH
        self._roach = corr.katcp_wrapper.FpgaClient(settings.feng_configuration.roach_name)

        # Check if connection was successful
        if not self._roach.is_connected:
            try:
                self._roach.stop()
            except BaseException:
                pass
            raise Exception("BiralesBackend: Could not connect to ROACH")

        # Antenna map (ordered per cylinder from 1 to 8) and ADC map
        self._antenna_map = [0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27, 4, 12,
                             20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31]

        self._coefficients_order = [0, 8, 16, 24, 4, 12, 20, 28, 1, 9, 17, 25, 5, 13, 21, 29,
                                    2, 10, 18, 26, 6, 14, 22, 30, 3, 11, 19, 27, 7, 15, 23, 31]

    def start(self, program_fpga=False, equalize=False, calibrate=False):
        """ Start the ROACH-II backend """
        # Check if roach is connected
        if not self._roach.is_connected:
            raise Exception("BiralesBackend: Cannot start backend on unconnected roach")

        # Check if roach is already programmed
        if self._roach.status() == "program" and self.read_startup_time() != 0:
            logging.info("ROACH already set up, skipping start")
            return

        # Get configuration
        config_files = settings.roach_config_files
        config = settings.feng_configuration

        # Process header file
        header = self._get_header_keys(config_files.header_file)

        # If required, load bitstream and initialise control software register to 0
        while True:
            if program_fpga:
                logging.info("Programming roach")
                self._program_roach(config.bitstream)
                self._roach.write_int('ctrl_sw', 0)
            else:
                logging.info("Skipping programming")

            # Write base configuration
            self._write_base_header(config, header)

            # Set number of antennas
            self._roach.write_int('n_ant', len(settings.beamformer.antenna_locations))

            logging.info("Setting frequency channel to %d".format(settings.roach_observation.freq_channel))
            self._roach.write_int('channel1', settings.roach_observation.freq_channel)

            # Set ADC map
            adc_map = '----------------------------------------------------------------'
            adc_map += '----------------------------------------------------------------'
            antennas = []
            with open(config_files.antenna_order, "r") as f:
                ant_list = f.readlines()
                for i in xrange(len(ant_list)):
                    antennas += [ant_list[i].split()]
                    antennas[i][1] = self._antenna_map[int(antennas[i][1])]
                    adc_map = adc_map[:antennas[i][1] * 4] + antennas[i][0] + adc_map[antennas[i][1] * 4 + 4:]
                    self._roach.write('ant_list', struct.pack('>I', antennas[i][1]), i * 4)

            self._write_header('adc_map', adc_map, header)

            # Set up 10GbE interface for data transmission
            gbe_0 = getattr(config, 'gbe-0')
            gbe_0_dest_ip = getattr(config, 'gbe-0_dest_ip')
            gbe_0_dest_port = getattr(config, 'gbe-0_dest_port')
            gbe_0_pkt_len = getattr(config, 'gbe-0_pkt_len')
            logging.info("Starting interface %s" % gbe_0)
            self._roach.write_int(gbe_0 + '_destip', gbe_0_dest_ip)
            time.sleep(0.3)
            self._roach.write_int(gbe_0 + '_destport', gbe_0_dest_port)
            time.sleep(0.3)
            self._roach.write_int(gbe_0 + '_len', gbe_0_pkt_len)
            time.sleep(0.3)
            ipconv = str(int((gbe_0_dest_ip & 255 * 256 * 256 * 256) >> 24))
            ipconv += "." + str(int((gbe_0_dest_ip & 255 * 256 * 256) >> 16))
            ipconv += "." + str(int((gbe_0_dest_ip & 255 * 256) >> 8))
            ipconv += "." + str(int(gbe_0_dest_ip & 255))

            logging.info("Set UDP packets destination IP:Port to %s:%d" % (ipconv, gbe_0_dest_port))
            ip = 3232238524
            mac = (0 << 40) + (96 << 32) + ip
            self._roach.tap_start('tap0', gbe_0, mac, ip, gbe_0_dest_port)
            time.sleep(0.3)
            logging.info("UDP packets started")

            # Set FFT shift
            time.sleep(0.1)
            self._change_ctrl_sw_bits(0, 10, config.fft_shift)

            # Perform amplitude equalization if required
            if equalize:
                logging.info("Performing amplitude equalization")
                self._change_ctrl_sw_bits(20, 20, 1)
            else:
                logging.info("Skipping amplitude equalization")
                self._change_ctrl_sw_bits(20, 20, 0)

            # Calibrating ADCs
            logging.info("Calibrating ADCs")
            self._adc_cal()
            time.sleep(0.05)
            self._roach.write_int('adc_spi_ctrl', 1)
            time.sleep(.05)
            self._roach.write_int('adc_spi_ctrl', 0)
            time.sleep(.05)
            for i in range(5):
                time.sleep(0.5)
                self._check_adc_sync()

            # Arm the F-Engine
            logging.info("Arming F Engine and setting FFT Shift")
            trig_time = self._feng_arm()
            logging.info('Armed. Expect trigger at %s local (%s UTC).' % (
                time.strftime('%H:%M:%S', time.localtime(trig_time)),
                time.strftime('%H:%M:%S', time.gmtime(trig_time))))
            logging.info("Updating header BRAM with %s=%d" % ('t_zero', trig_time))
            self._write_header('t_zero', trig_time, header)
            logging.info("Read from header t_zero=%d" % (self._read_header('t_zero', header)))
            logging.info("Updating header BRAM with %s=%d" % ('fft_shift', config.fft_shift))
            self._write_header('fft_shift', config.fft_shift, header)
            logging.info("Read from header fft_shift=%d" % (self._read_header('fft_shift', header)))

            self._arm_sync()

            logging.info("Verifying ADC signals...please hold on")
            bit_ptp = self._adc_bit_ptp()
            if bit_ptp < 1:
                logging.info("FPGA <--> ADC sync OK (adc_bit_ptp reporting %3.1f)" % bit_ptp)
                break
            logging.warning("FPGA <--> ADC sync NOT OK (adc_bit_ptp reporting %3.1f), retrying..." % bit_ptp)

        # Download calibration coefficients if required
        if calibrate:
            self.load_calibration_coefficients(config_files.amp_eq_file, config_files.phase_eq_file)

        # Reset interface
        logging.info("Resetting gbe interface")
        time.sleep(1)
        self._roach.write_int('gbe_reset', 0)
        self._roach.write_int('gbe_reset', 1)
        time.sleep(1)
        self._roach.write_int('gbe_reset', 0)
        self._roach.write_int('gbe_reset', 1)
        time.sleep(1)
        self._roach.write_int('gbe_reset', 0)
        self._roach.write_int('gbe_reset', 1)

        logging.info("Birales ROACH backend initialised")
    
    def stop(self):
        self._roach.stop()

    def load_calibration_coefficients(self, amplitude_filepath=None, phase_filepath=None,
                                      amplitude=None, phase=None):
        """
        Load coefficients

        :param amplitude_filepath:
        :param phase_filepath:
        :param amplitude:
        :param phase:
        :return:
        """

        # Load amplitude coefficients
        if amplitude_filepath is not None:
            with open(amplitude_filepath, 'r') as f:
                amplitude = {}
                for item in f.readlines():
                    c = item[:-1].split()
                    amplitude[c[0]] = float(c[1])
        else:
            # Coefficients coming from dictionaries, need to re-order
            new_amplitude = {}
            for k, v in amplitude.iteritems():
                new_amplitude[k] = self._coefficients_order[int(k.lstrip('a'))]
            amplitude = new_amplitude

        amplitude_coefficients = np.zeros([32, 1024 / 4], dtype=float)  # 32 antennas, 1024 chans
        values = []
        indices = []
        for k, val in amplitude.iteritems():
            index = int(k.lstrip('a'))
            indices += [index]
            values += [val]
            amplitude_coefficients[index, :] = val  # give all channels the same coeff
        amp_header = np.zeros([32], dtype=float)
        for a in range(len(indices)):
            amp_header[indices[a]] = values[a]

        # Load phase coefficients
        if phase_filepath is not None:
            with open(phase_filepath, 'r') as f:
                phase = {}
                for item in f.readlines():
                    c = item[:-1].split()
                    phase[c[0]] = float(c[1])
        else:
            # Coefficients coming from dictionaries, need to re-order
            new_phase = {}
            for k, v in phase.iteritems():
                new_phase[k] = self._coefficients_order[int(k.lstrip('a'))]
            phase = new_phase

        phs_coeffs = np.zeros([32, 1024 / 4], dtype=complex)  # 32 ants, 1024 chans
        values = []
        indices = []
        for k, val in phase.iteritems():
            index = int(k.lstrip('a'))
            value = np.exp(1j * val * np.pi / 180.)
            phs_coeffs[index, :] = value  # give all channels the same coeff
            indices += [index]
            values += [val]
        phs_header = np.zeros([32], dtype=float)  # 32 ants
        for a in range(len(indices)):
            phs_header[indices[a]] = values[a]

        # Map the antenna numberings used for the coefficients to the numberings used for the f-engine
        # here we assume they are the same
        ant_remap = np.arange(32)

        new_amp_coeffs = np.ones((32, 1, 512 / 4), dtype=float)
        new_phs_coeffs = np.ones((32, 1, 512 / 4), dtype=complex)
        for ant in range(32):
            new_amp_coeffs[ant] = self._modify_amplitude_coefficients(ant, 0,
                                                                      amplitude_coefficients[ant_remap[ant], ::-1],
                                                                      closed_loop=False)
            new_phs_coeffs[ant] = self._modify_phase_coefficients(ant, 0, phs_coeffs[ant_remap[ant], ::-1],
                                                                  closed_loop=False)

        logging.info("Writing phase coefficients")
        self._eq_write_all_phs(new_phs_coeffs)
        value = phs_header.tolist()
        val = ''
        for i in range(len(value)):
            val += struct.pack('>f', value[i])
        self._roach.write('header', val, 408)

        logging.info("Writing amplitude coefficients")
        self._eq_write_all_amp(new_amp_coeffs)
        value = amp_header.tolist()
        val = ''
        for i in range(len(value)):
            val += struct.pack('>f', value[i])

        self._roach.write('header', val, 280)

        logging.info("Finished loading calibration coefficients")

    def read_startup_time(self):
        """ Read ROACH startup time """
        if not self._roach.is_connected():
            return 0

        try:
            return struct.unpack(">I", self._roach.read('header', 4, 0))[0]
        except:
            return 0

    # --------------------------------------- Helper methods ------------------------------------

    def _program_roach(self, bitstream):
        """ Load bistream to ROACH"""
        logging.info('Deprogramming FPGAs')
        self._roach.progdev('')
        time.sleep(0.1)
        logging.info('Programming with bitstream %s' % (bitstream))
        self._roach.progdev(bitstream)
        time.sleep(0.1)

    def _change_ctrl_sw_bits(self, lsb, msb, val, ctrl='ctrl_sw'):
        """

        :param lsb:
        :param msb:
        :param val:
        :param ctrl:
        :return:
        """

        # Change control software bits
        num_bits = msb - lsb + 1
        if val > (2 ** num_bits - 1):
            raise ValueError("ERROR: Attempting to write value to ctrl_sw which exceeds available bit width")

        # Create a mask which has value 0 over the bits to be changed
        mask = (2 ** 32 - 1) - ((2 ** num_bits - 1) << lsb)

        # Remove the current value stored in the ctrl_sw bits to be changed
        ctrl_sw_value = self._roach.read_uint(ctrl)
        ctrl_sw_value &= mask

        # Insert the new value
        ctrl_sw_value = ctrl_sw_value + (val << lsb)

        # Write
        self._roach.write_int(ctrl, ctrl_sw_value)

    def _check_adc_sync(self):
        """ Check ADC sychronization"""
        rv = self._roach.read_uint('adc_sync_test')
        while (rv & 0b111) != 1:
            self._roach.write_int('adc_spi_ctrl', 1)
            time.sleep(.05)
            self._roach.write_int('adc_spi_ctrl', 0)
            time.sleep(.05)
            logging.warn("ADC sync test returns %i" % rv)
            rv = self._roach.read_uint('adc_sync_test')

        logging.info("ADC sync test returns %i (1 = ADC syncs present & aligned)" % rv)

    def _feng_arm(self):
        """ Arms all F engines, records arm time in config file and issues SPEAD update. Returns the UTC time at
        which the system was sync'd in seconds since the Unix epoch (MCNT=0) """
        # Wait for within 100ms of a half-second, then send out the arm signal.
        ready = (int(time.time() * 10) % 5) == 0
        while not ready:
            ready = (int(time.time() * 10) % 5) == 0
        trig_time = time.time()
        self._arm_sync()  # Implicitly affects all FPGAs
        self._send_sync()
        return int(trig_time)

    def _arm_sync(self):
        self._change_ctrl_sw_bits(11, 11, 0)
        self._change_ctrl_sw_bits(11, 11, 1)

    def _send_sync(self):
        self._change_ctrl_sw_bits(12, 12, 0)
        self._change_ctrl_sw_bits(12, 12, 1)

    def _adc_cal(self, calreg='x64_adc_ctrl'):
        """ Calibrate ADCs """
        DELAY_CTRL = 0x4
        DATASEL = 0x8
        DATAVAL = 0xc

        # Loop over all ADCs
        for j in range(0, 8):
            # Select bit and reset dll
            self._roach.blindwrite(calreg, '%c%c%c%c' % (0x0, 0x0, 0x0, j // 2), DATASEL)
            self._roach.blindwrite(calreg, '%c%c%c%c' % (0x0, 0x0, 0x0, (1 << j)), DELAY_CTRL)

            stable = 1
            prev_val = 0
            while stable == 1:
                self._roach.blindwrite(calreg, '%c%c%c%c' % (0x0, 0xff, (1 << j), 0x0), DELAY_CTRL)
                val = struct.unpack('>L', (self._roach.read(calreg, 4, DATAVAL)))[0]
                val0 = (val & (0xffff << (16 * (j % 2)))) >> (16 * (j % 2))
                stable = (val0 & 0x1000) >> 12
                if val0 != prev_val and prev_val != 0:
                    break
                prev_val = val0
            for i in range(10):
                self._roach.blindwrite(calreg, '%c%c%c%c' % (0x0, 0xff, (1 << j), 0x0), DELAY_CTRL)

    def _write_base_header(self, config, header):
        """ Write base header configuration """
        for k, v in config.__dict__.iteritems():
            if k in ["__len__", "adc_curve", "header", "bitstream", "adc_debug", "roach_name", "katcp_port"]:
                continue
            self._write_header(k, v, header)

    def _write_header(self, field, value, header):
        """ Write header value """
        record = self._head_get_info(field, header)
        to_write = 0
        if record[3] == 'num':
            to_write = struct.pack('>I', value)
        if record[3] == 'numarr':
            to_write = struct.pack('>32L', value)
        if record[3] == 'str':
            to_write = str(value).ljust(record[1])

        self._roach.write("header", to_write, record[0])

    def _read_header(self, field, header):
        """ Read header value"""
        record = self._head_get_info(field, header)
        value = 0
        if record[3] == 'num':
            value = struct.unpack('>I', self._roach.read('header', 4, record[0]))[0]
        if record[3] == 'numarr':
            value = struct.unpack('>32L', self._roach.read('header', 128, record[0]))
        if record[3] == 'str':
            value = self._roach.read('header', record[1], record[0])
        return value

    @staticmethod
    def _head_get_info(key, header):
        for i in range(len(header)):
            if header[i][2] == key:
                return header[i]

        raise Exception('RoachBackend: Key Error on Header: %s' % key)

    @staticmethod
    def _get_header_keys(header_file):
        """ Extract header keys from header file """
        header = []
        f_head = open(header_file)
        head_list = f_head.readlines()
        for i in range(len(head_list))[2:]:
            header += [head_list[i].split('#')[0].split('\t')[:-1]]
            header[i - 2][0] = int(header[i - 2][0])
            header[i - 2][1] = int(header[i - 2][1])
        f_head.close()
        return header

    @staticmethod
    def _modify_amplitude_coefficients(antenna, pol, coefficients, closed_loop=True):
        """Multiply the current coefficients by a constant, or a vector, or replace them with a new calibration set"""
        nof_channels = 1024 / 4
        nof_antennas = 32
        nof_polarisations = 1
        decimation = 2
        nof_coefficients = nof_channels / decimation
        coeff = np.ones((nof_antennas, nof_polarisations, nof_channels / decimation), dtype=float)
        if np.size(coefficients) == 1:
            coefficients = np.array([coefficients], dtype=float)
        if len(coefficients) == len(coeff[antenna, pol]):
            # Check the number of calibration coefficients is the same as the number of coefficients already
            # associated with the manager instance
            dec_coeffs = np.array(coefficients, dtype=float)
        elif len(coefficients) == nof_channels:
            # if it isn't, but there are the same number of calibration coeffs as there are channels, then decimate
            # and apply the calibration
            dec_coeffs = np.zeros_like(coeff[antenna, pol])
            for i in range(nof_coefficients):
                dec_coeffs[i] = np.average(coefficients[i * decimation:(i + 1) * decimation])
        elif len(coefficients) == 1:
            # if there's only one calibration coefficient, apply it to all the channels
            dec_coeffs = np.ones_like(coeff[antenna, pol]) * coefficients[0]
        else:
            raise IndexError('''The number of calibration coefficients don\'t seem to be the number
                             or frequency channels, the number of decimated channels or a single value. I have
                             no idea what to do!''')
        if not closed_loop:
            coeff[antenna][pol] = dec_coeffs
        else:
            coeff[antenna, pol] = coeff[antenna, pol] * dec_coeffs
        return coeff[antenna]

    @staticmethod
    def _modify_phase_coefficients(ant, pol, coefficients, closed_loop=True):
        """Multiply the current coefficients by a constant, or a vector, or replace them with a new calibration set"""
        nof_channels = 1024 / 4
        nof_antennas = 32
        nof_polariations = 1
        decimation = 2
        nof_coefficients = nof_channels / decimation
        coeff = np.ones((nof_antennas, nof_polariations, nof_channels / decimation), dtype=complex)
        if np.size(coefficients) == 1:
            coefficients = np.array([coefficients], dtype=complex)
        if len(coefficients) == len(coeff[ant, pol]):
            # Check the number of calibration coefficients is the same as the number of coefficients already
            # associated with the manager instance
            dec_coeffs = np.array(coefficients, dtype=complex)
        elif len(coefficients) == nof_channels:
            # if it isn't, but there are the same number of calibration coeffs as there are channels, then decimate
            # and apply the calibration
            dec_coeffs = np.zeros_like(coeff[ant, pol])
            for i in range(nof_coefficients):
                dec_coeffs[i] = np.average(coefficients[i * decimation:(i + 1) * decimation])
        elif len(coefficients) == 1:
            # if there's only one calibration coefficient, apply it to all the channels
            dec_coeffs = np.ones_like(coeff[ant, pol]) * coefficients[0]
        else:
            raise IndexError('''The number of calibration coefficients don\'t seem to be the number
                             or frequency channels, the number of decimated channels or a single value. I have
                             no idea what to do!''')
        if closed_loop is False:
            coeff[ant][pol] = dec_coeffs
        else:
            coeff[ant, pol] = coeff[ant, pol] * dec_coeffs
        return coeff[ant]

    def _eq_write_all_amp(self, new_amp_coeffs):
        """Write to Amplitude BRAM the equalization coefficents for a given antpol on the F Engine"""
        coeffs = self._get_real_fp(new_amp_coeffs, 32, 16, signed=False)
        uints = np.array(coeffs, dtype=np.uint32)
        MAP = [0, 4, 1, 5, 2, 6, 3, 7]

        for pn, pol in enumerate(['x', 'y'][0:1]):
            for eq_subsys in range(32 // 8):
                bin_str = ''
                for ant_mux_index in range(8):
                    bin_str += np.array(uints[MAP[ant_mux_index] + 8 * eq_subsys, pn], dtype='>u4').tostring()
                self._roach.write('amp_EQ%d_coeff_bram' % eq_subsys, bin_str)
                time.sleep(0.2)

    def _eq_write_all_phs(self, new_phs_coeffs):
        """Write to Phase BRAM the equalization coefficents for a given antpol on the F Engine"""
        coeffs = self._get_complex_fp(new_phs_coeffs, 16, 15)
        uints = ((np.array(coeffs.real, dtype=int) & 0xffff) << 16) + (np.array(coeffs.imag, dtype=int) & 0xffff)
        MAP = [0, 4, 1, 5, 2, 6, 3, 7]
        for pn, pol in enumerate(['x', 'y'][0:1]):
            for eq_subsys in range(32 // 8):
                bin_str = ''
                for ant_mux_index in range(8):
                    bin_str += np.array(uints[MAP[ant_mux_index] + 8 * eq_subsys, pn], dtype='>u4').tostring()
                self._roach.write('phase_EQ%d_coeff_bram' % eq_subsys, bin_str)
                time.sleep(0.2)

    @staticmethod
    def _get_complex_fp(coeff, bitwidth, bp, signed=True):
        if signed:
            clipbits = bitwidth - 1
        else:
            clipbits = bitwidth
        real = np.clip(np.round(np.real(coeff) * 2 ** bp), -2 ** clipbits - 1, 2 ** clipbits - 1)
        imag = np.clip(np.round(np.imag(coeff) * 2 ** bp), -2 ** clipbits - 1, 2 ** clipbits - 1)
        return np.array(real + 1j * imag)

    @staticmethod
    def _get_real_fp(coeff, bitwidth, bp, signed=False):
        if signed:
            clipbits = bitwidth - 1
        else:
            clipbits = bitwidth
        return np.clip(np.round(np.real(coeff) * 2 ** bp), -2 ** clipbits - 1, 2 ** clipbits - 1)

    def _get_adc_power(self, antenna):
        adc_levels_acc_len = 32
        adc_bits = 12

        self._roach.write_int('adc_sw_adc_sel', antenna)
        time.sleep(.05)
        rv = self._roach.read_uint('adc_sw_adc_sum_sq')

        pwrX = float(rv)
        rmsX = np.sqrt(pwrX / adc_levels_acc_len) / (2 ** (adc_bits - 1))
        bitsX = max(np.log2(rmsX * (2 ** adc_bits)), 0.)

        return rmsX * 2 ** (adc_bits - 1), bitsX

    def _adc_bit_ptp(self):
        allin = []
        for i in range(len(settings.beamformer.antenna_locations)):
            rmsA, powA = self._get_adc_power(i)
            allin += [powA]
        return np.ptp(allin)

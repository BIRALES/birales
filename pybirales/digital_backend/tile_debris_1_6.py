import functools
import logging
import socket
import numpy as np
import time
import math
import os

from pyfabil.base.definitions import *
from pyfabil.base.utils import ip2long
from pyfabil.boards.tpm_1_6 import TPM_1_6
from pybirales.digital_backend.tile_debris import Tile

# Helper to disallow certain function calls on unconnected tiles
def connected(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if self.tpm is None:
            self.logger.warn("Cannot call function {} on unconnected TPM".format(f.__name__))
            raise LibraryError("Cannot call function {} on unconnected TPM".format(f.__name__))
        else:
            return f(self, *args, **kwargs)

    return wrapper


class Tile_1_6(Tile):
    def __init__(self, ip="10.0.10.2", port=10000, lmc_ip="10.0.10.1", lmc_port=4660,
                 sampling_rate=800e6, ddc_frequency=139.65e6):

        super(Tile_1_6, self).__init__(
            ip, port, lmc_ip, lmc_port, sampling_rate, ddc_frequency
        )

        self._decimation_ratio = 8
        self._sampling_rate = sampling_rate

        nco_freq = int(ddc_frequency / sampling_rate * 2**48)
        self._ddc_frequency = float(nco_freq) / 2**48 * sampling_rate

        self.daq_modes_with_timestamp_flag = ["raw_adc_mode", "channelized_mode", "beamformed_mode"]

    # ---------------------------- Main functions ------------------------------------
    def tpm_version(self):
        """
        Determine whether this is a TPM V1.2 or TPM V1.6
        :return: TPM hardware version
        :rtype: string
        """
        return "tpm_v1_6"

    def connect(self, initialise=False, simulation=False, enable_ada=False):

        # Try to connect to board, if it fails then set tpm to None
        self.tpm = TPM_1_6()

        # Add plugin directory (load module locally)
        tf = __import__("pybirales.digital_backend.plugins.tpm_1_6.tpm_debris_firmware", fromlist=[None])
        self.tpm.add_plugin_directory(os.path.dirname(tf.__file__))

        self.tpm.connect(ip=self._ip,
                         port=self._port,
                         initialise=initialise,
                         simulator=simulation,
                         enable_ada=enable_ada,
                         fsample=self._sampling_rate,
                         ddc=True,
                         fddc=self._ddc_frequency,
                         adc_low_bitrate=0x1)

        # Load tpm debris firmware for both FPGAs (no need to load in simulation)
        if not simulation and self.tpm.is_programmed():
            self.tpm.load_plugin("Tpm_1_6_DebrisFirmware", device=Device.FPGA_1, fsample=self._sampling_rate,
                                 fddc=self._ddc_frequency, decimation=self._decimation_ratio)
            self.tpm.load_plugin("Tpm_1_6_DebrisFirmware", device=Device.FPGA_2, fsample=self._sampling_rate,
                                 fddc=self._ddc_frequency, decimation=self._decimation_ratio)
        elif not self.tpm.is_programmed():
            self.logger.warn("TPM is not programmed! No plugins loaded")

    def initialise(self, enable_ada=False, enable_test=False):
        """ Connect and initialise """

        # Connect to board
        self.connect(initialise=True, enable_ada=enable_ada)

        # Before initialing, check if TPM is programmed
        if not self.tpm.is_programmed():
            self.logger.error("Cannot initialise board which is not programmed")
            return

        # Disable debug UDP header
        self["board.regfile.ena_header"] = 0x1

        # Initialise firmware plugin
        for firmware in self.tpm.tpm_debris_firmware:
            firmware.initialise_firmware()

        self['fpga1.dsp_regfile.spead_tx_enable'] = 1
        self['fpga2.dsp_regfile.spead_tx_enable'] = 1

        # Setting shutdown temperature in the CPLD
        # self.tpm.set_shutdown_temperature(60)

        # Set LMC IP
        self.tpm.set_lmc_ip(self._lmc_ip, self._lmc_port)

        # Enable C2C streaming
        self.tpm["board.regfile.ena_stream"] = 0x0
        self.tpm["board.regfile.ena_stream"] = 0x1
        self.set_c2c_burst()

        # Synchronise FPGAs
        self.sync_fpga_time()

        # Initialise ADAs
        # self.tpm.tpm_ada.initialise_adas()

        # Reset test pattern generator
        self.tpm.test_generator[0].channel_select(0x0000)
        self.tpm.test_generator[1].channel_select(0x0000)
        self.tpm.test_generator[0].disable_prdg()
        self.tpm.test_generator[1].disable_prdg()

        # Use test_generator plugin instead!
        if enable_test:
            # Test pattern. Tones on channels 72 & 75 + pseudo-random noise
            self.logger.info("Enabling test pattern")
            for generator in self.tpm.test_generator:
                generator.set_tone(0, 72 * self._sampling_rate / 1024, 0.0)
                generator.enable_prdg(0.4)
                generator.channel_select(0xFFFF)

        # Set destination and source IP/MAC/ports for 10G cores
        # This will create a loopback between the two FPGAs
        # ip_octets = self._ip.split('.')
        # for n in range(8):
        #     src_ip = "10.{}.{}.{}".format(ip_octets[2], n + 1, ip_octets[3])
        #     dst_ip = "10.{}.{}.{}".format(ip_octets[2], (1 + n) + (4 if n < 4 else -4), ip_octets[3])
        #     self.configure_10g_core(n,
        #                             src_mac=0x620000000000 + ip2long(src_ip),
        #                             dst_mac=0x620000000000 + ip2long(dst_ip),
        #                             src_ip=src_ip,
        #                             dst_ip=dst_ip,
        #                             src_port=0xF0D0,
        #                             dst_port=4660)
        #
        # # wait UDP link up
        # self.logger.info("Waiting for 10G link...")
        # try:
        #     times = 0
        #     while True:
        #         linkup = 1
        #         for n in [0, 1, 2, 4, 5, 6]:
        #             core_status = self.tpm.tpm_10g_core[n].get_arp_table_status(0, silent_mode=True)
        #             if core_status & 0x4 == 0:
        #                 linkup = 0
        #         if linkup == 1:
        #             self.logger.info("10G Link established! ARP table populated!")
        #             break
        #         else:
        #             times += 1
        #             time.sleep(0.5)
        #             if times == 20:
        #                 self.logger.warning("10G Links not established after 10 seconds! ARP table not populated!")
        #                 break
        # except:
        #     time.sleep(4)
        #     self.mii_exec_test(10, False)
        #     self['fpga1.regfile.eth10g_ctrl'] = 0x0
        #     self['fpga2.regfile.eth10g_ctrl'] = 0x0

        self.configure_channeliser()

    def is_qsfp_module_plugged(self, qsfp_id=0):
        """
        Initialise firmware components.

        :return: True when cable is detected
        """
        qsfp_status = self.tpm.tpm_qsfp_adapter[qsfp_id].get('ModPrsL')
        if qsfp_status == 0:
            return True
        else:
            return False



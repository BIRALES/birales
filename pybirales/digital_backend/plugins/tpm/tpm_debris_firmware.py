from math import ceil

__author__ = 'Alessio Magro'

import logging
import time

from pyfabil.plugins.firmwareblock import FirmwareBlock
from pyfabil.base.definitions import *
from time import sleep
from math import log, ceil


class TpmDebrisFirmware(FirmwareBlock):
    """ FirmwareBlock tests class """

    @firmware({'design': 'tpm_debris', 'major': '0', 'minor': '0'})
    @compatibleboards(BoardMake.TpmBoard)
    @friendlyname('tpm_debris_firmware')
    @maxinstances(2)
    def __init__(self, board, **kwargs):
        """ TpmDebrisFirmware initializer
        :param board: Pointer to board instance
        """
        super(TpmDebrisFirmware, self).__init__(board)

        # Device must be specified in kwargs
        if kwargs.get("device", None) is None:
            raise PluginError("TpmDebrisFirmware requires device argument")
        self._device = kwargs["device"]

        if kwargs.get("fsample", None) is None:
            logging.info("TpmDebrisFirmware: Setting default sampling frequency 800 MHz.")
            self._fsample = 800e6
        else:
            self._fsample = float(kwargs["fsample"])

        if kwargs.get('fddc', None) is None:
            logging.info("TpmDebrisFirmware: Setting default DDC frequency 139.65 MHz.")
            self._fddc = 139.65
        else:
            self._fddc = float(kwargs['fddc'])

        if kwargs.get('decimation', None) is None:
            logging.info("TpmDebrisFirmware: Setting default decimation 8.")
            self._decimation = 8
        else:
            self._decimation = float(kwargs['decimation'])

        try:
            if self.board['fpga1.regfile.feature.xg_eth_implemented'] == 1:
                self.xg_eth = True
            else:
                self.xg_eth = False
            if self.board['fpga1.regfile.feature.xg_eth_40g_implemented'] == 1:
                self.xg_40g_eth = True
            else:
                self.xg_40g_eth = False
        except:
            self.xg_eth = False
            self.xg_40g_eth = False

        # Load required plugins
        self._jesd1 = self.board.load_plugin("TpmJesd", device=self._device, core=0, frame_length=1024)
        self._jesd2 = self.board.load_plugin("TpmJesd", device=self._device, core=1, frame_length=1024)
        self._fpga = self.board.load_plugin('TpmFpga', device=self._device)
        if self.xg_eth and self.xg_40g_eth:
            self._fortyg = self.board.load_plugin("TpmFortyGCoreXg", device=self._device, core=0)
        self._testgen = self.board.load_plugin("TpmTestGenerator", device=self._device,
                                               fsample=self._fsample / self._decimation)
        self._sysmon = self.board.load_plugin("TpmSysmon", device=self._device)
        self._patterngen = self.board.load_plugin("TpmPatternGenerator", device=self._device)
        self._power_meter = self.board.load_plugin("AdcPowerMeterSimple",
                                                   device=self._device,
                                                   fsample=self._fsample / self._decimation,
                                                   samples_per_frame=1024)
        self._fast_detect_statistics = self.board.load_plugin("FastDetectStatistics",
                                                              device=self._device,
                                                              fsample=self._fsample / self._decimation,
                                                              samples_per_frame=1024)
        self._integrator = self.board.load_plugin("TpmIntegrator", device=self._device,
                                                  fsample=self._fsample / self._decimation, nof_frequency_channels=512,
                                                  oversampling_factor=1.0)
        self._polyfilter = self.board.load_plugin("PolyFilter", device=self._device)

        self._device_name = "fpga1" if self._device is Device.FPGA_1 else "fpga2"

    def initialise_firmware(self):
        """ Initialise firmware components """
        max_retries = 4
        retries = 0

        while True:
            self._fpga.fpga_global_reset()

            self._fpga.fpga_mmcm_config(self._fsample / 2)
            self._fpga.fpga_jesd_gth_config(self._fsample / 2)  # GTH are configured for 4 Gbps

            self._fpga.fpga_reset()

            # Start JESD cores
            self._jesd1.jesd_core_start(single_lane=True)
            self._jesd2.jesd_core_start(single_lane=True)

            # Initialise FPGAs
            # I have no idea what these ranges are
            self._fpga.fpga_start(range(16), range(16))

            # Check if it's correct
            if self.board['%s.jesd204_if.regfile_status' % self._device_name] & 0x1F == 0x1E:
                break

            retries += 1
            sleep(0.2)

            if retries == max_retries:
                raise BoardError("TpmDebrisFirmware: Could not configure JESD cores")

        # Initialise power meter
        self._power_meter.initialise()
        self._fast_detect_statistics.set_integration_time(1.0)

        # Initialise 10G cores
        # for teng in self._teng:
        #    teng.initialise_core()

        self._patterngen.initialise()

    #######################################################################################

    def send_raw_data(self):
        """ Send raw data from the TPM """
        self.board["%s.lmc_gen.raw_all_channel_mode_enable" % self._device_name] = 0x0
        self.board["%s.lmc_gen.request.raw_data" % self._device_name] = 0x1

    def send_raw_data_synchronised(self):
        """ Send raw data from the TPM """
        self.board["%s.lmc_gen.raw_all_channel_mode_enable" % self._device_name] = 0x1
        self.board["%s.lmc_gen.request.raw_data" % self._device_name] = 0x1

    def clear_lmc_data_request(self):
        """Stop transmission of all LMC data."""
        self.board[self._device_name + ".lmc_gen.request"] = 0

    def stop_integrated_channel_data(self):
        """ Stop receiving integrated beam data from the board """
        self._integrator.stop_integrated_channel_data()

    def stop_integrated_data(self):
        """ Stop transmission of integrated data"""
        self._integrator.stop_integrated_data()

    ##################### Superclass method implementations #################################

    def initialise(self):
        """ Initialise TpmDebrisFirmware """
        logging.info("TpmDebrisFirmware has been initialised")
        return True

    def status_check(self):
        """ Perform status check
        :return: Status
        """
        logging.info("TpmDebrisFirmware : Checking status")
        return Status.OK

    def clean_up(self):
        """ Perform cleanup
        :return: Success
        """
        logging.info("TpmDebrisFirmware : Cleaning up")
        return True

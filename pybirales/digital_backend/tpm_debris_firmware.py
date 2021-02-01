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
        if kwargs.get('device', None) is None:
            raise PluginError("TpmDebrisFirmware requires device argument")
        self._device = kwargs['device']

        if kwargs.get('fsample', None) is None:
            logging.info("TpmDebrisFirmware: Setting default sampling frequency 800 MHz.")
            self._fsample = 800e6
        else:
            self._fsample = float(kwargs['fsample'])

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
        self._jesd1 = self.board.load_plugin("TpmJesd", device=self._device, core=0)
        self._jesd2 = self.board.load_plugin("TpmJesd", device=self._device, core=1)
        self._fpga = self.board.load_plugin('TpmFpga', device=self._device)
        if self.xg_eth and not self.xg_40g_eth:
            self._teng = [self.board.load_plugin("TpmTenGCoreXg", device=self._device, core=0),
                          self.board.load_plugin("TpmTenGCoreXg", device=self._device, core=1),
                          self.board.load_plugin("TpmTenGCoreXg", device=self._device, core=2),
                          self.board.load_plugin("TpmTenGCoreXg", device=self._device, core=3)]
        elif self.xg_eth and self.xg_40g_eth:
            self._fortyg = self.board.load_plugin("TpmFortyGCoreXg", device=self._device, core=0)
        else:
            self._teng = [self.board.load_plugin("TpmTenGCore", device=self._device, core=0),
                          self.board.load_plugin("TpmTenGCore", device=self._device, core=1),
                          self.board.load_plugin("TpmTenGCore", device=self._device, core=2),
                          self.board.load_plugin("TpmTenGCore", device=self._device, core=3)]
        self._testgen = self.board.load_plugin("TpmTestGenerator", device=self._device, fsample=self._fsample/self._decimation)
        self._sysmon = self.board.load_plugin("TpmSysmon", device=self._device)
        self._patterngen = self.board.load_plugin("TpmPatternGenerator", device=self._device)
        self._power_meter = self.board.load_plugin("AdcPowerMeterSimple", device=self._device, fsample=self._fsample/(self._decimation), samples_per_frame=4096)
        self._integrator = self.board.load_plugin("TpmIntegrator", device=self._device, fsample=self._fsample/self._decimation, nof_frequency_channels=2048, oversampling_factor=1.0)
        self._polyfilter = self.board.load_plugin("PolyFilter", device=self._device)

        self._device_name = "fpga1" if self._device is Device.FPGA_1 else "fpga2"

    def fpga_clk_sync(self):
        """ FPGA synchronise clock"""

        if self._device_name == 'fpga1':

            fpga0_phase = self.board['fpga1.pps_manager.sync_status.cnt_hf_pps']

            # restore previous counters status using PPS phase
            self.board['fpga1.pps_manager.sync_tc.cnt_1_pulse'] = 0
            time.sleep(1.1)
            for n in range(5):
                fpga0_cnt_hf_pps = self.board['fpga1.pps_manager.sync_phase.cnt_hf_pps']
                if abs(fpga0_cnt_hf_pps - fpga0_phase) <= 3:
                    logging.debug("FPGA1 clock synced to PPS phase!")
                    break
                else:
                    rd = self.board['fpga1.pps_manager.sync_tc.cnt_1_pulse']
                    self.board['fpga1.pps_manager.sync_tc.cnt_1_pulse'] = rd + 1
                    time.sleep(1.1)

        if self._device_name == 'fpga2':

            # Synchronize FPGA2 to FPGA1 using sysref phase
            fpga0_phase = self.board['fpga1.pps_manager.sync_phase.cnt_1_sysref']

            self.board['fpga2.pps_manager.sync_tc.cnt_1_pulse'] = 0x0
            sleep(0.1)
            for n in range(5):
                fpga1_phase = self.board['fpga2.pps_manager.sync_phase.cnt_1_sysref']
                if fpga0_phase == fpga1_phase:
                    logging.debug("FPGA2 clock synced to SYSREF phase!")
                    break
                else:
                    rd = self.board['fpga2.pps_manager.sync_tc.cnt_1_pulse']
                    self.board['fpga2.pps_manager.sync_tc.cnt_1_pulse'] = rd + 1
                    sleep(0.1)

            logging.debug("FPGA1 clock phase before adc_clk alignment: " + hex(self.board['fpga1.pps_manager.sync_phase']))
            logging.debug("FPGA2 clock phase before adc_clk alignment: " + hex(self.board['fpga2.pps_manager.sync_phase']))

    def initialise_firmware(self):
        """ Initialise firmware components """
        max_retries = 4
        retries = 0

        while True:
            self._fpga.fpga_global_reset()

            self._fpga.fpga_mmcm_config(self._fsample / 2)  # (self._fsample)  # generate 100 MHz ADC clock
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

        # Initialise 10G cores
        for teng in self._teng:
            teng.initialise_core()

    #######################################################################################

    def send_raw_data(self):
        """ Send raw data from the TPM """
        self.board["%s.lmc_gen.raw_all_channel_mode_enable" % self._device_name] = 0x0
        self.board["%s.lmc_gen.request.raw_data" % self._device_name] = 0x1

    def send_raw_data_synchronised(self):
        """ Send raw data from the TPM """
        self.board["%s.lmc_gen.raw_all_channel_mode_enable" % self._device_name] = 0x1
        self.board["%s.lmc_gen.request.raw_data" % self._device_name] = 0x1

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

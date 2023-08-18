import functools
import logging
import socket
import numpy as np
import time
import math
import os

from pyfabil.base.definitions import *
from pyfabil.boards.tpm import TPM

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


class Tile(object):
    def __init__(self, ip="10.0.10.2", port=10000, lmc_ip="10.0.10.1", lmc_port=4660,
                 sampling_rate=800e6, ddc_frequency=139.65e6):
        self._lmc_port = lmc_port
        self._lmc_ip = socket.gethostbyname(lmc_ip)
        self._port = port
        self._ip = socket.gethostbyname(ip)
        self.tpm = None
        self.logger = logging

        self.station_id = 0
        self.tile_id = 0

        self._sampling_rate = sampling_rate
        self._decimation_ratio = 8

        nco_freq = int(ddc_frequency / sampling_rate * 4096.0)
        self._ddc_frequency = float(nco_freq) / 4096.0 * sampling_rate

        self.daq_modes_with_timestamp_flag = ["raw_adc_mode", "channelized_mode", "beamformed_mode"]

    # ---------------------------- Main functions ------------------------------------
    def tpm_version(self):
        """
        Determine whether this is a TPM V1.2 or TPM V1.6
        :return: TPM hardware version
        :rtype: string
        """
        return "tpm_v1_2"

    def connect(self, initialise=False, simulation=False, enable_ada=False):

        # Try to connect to board, if it fails then set tpm to None
        self.tpm = TPM()

        # Add plugin directory (load module locally)
        tf = __import__("pybirales.digital_backend.plugins.tpm.tpm_debris_firmware", fromlist=[None])
        self.tpm.add_plugin_directory(os.path.dirname(tf.__file__))

        self.tpm.connect(ip=self._ip,
                         port=self._port,
                         initialise=initialise,
                         simulator=simulation,
                         enable_ada=enable_ada,
                         fsample=self._sampling_rate,
                         ddc=True,
                         fddc=self._ddc_frequency,
                         adc_low_bitrate=True)

        # Load tpm debris firmware for both FPGAs (no need to load in simulation)
        if not simulation and self.tpm.is_programmed():
            self.tpm.load_plugin("TpmDebrisFirmware", device=Device.FPGA_1, fsample=self._sampling_rate,
                                 fddc=self._ddc_frequency, decimation=self._decimation_ratio)
            self.tpm.load_plugin("TpmDebrisFirmware", device=Device.FPGA_2, fsample=self._sampling_rate,
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

        if self['board.regfile.date_code'] < 0x19061700:
            self.logger.error("CPLD Firmware is too old. Minimum required version is 0x19061700")
            return

        # Disable debug UDP header
        self['board.regfile.header_config'] = 0x2


        # Initialise firmware plugin
        for firmware in self.tpm.tpm_debris_firmware:
            firmware.initialise_firmware()

        self['fpga1.dsp_regfile.spead_tx_enable'] = 1
        self['fpga2.dsp_regfile.spead_tx_enable'] = 1

        # Setting shutdown temperature in the CPLD
        self.tpm.set_shutdown_temperature(60)

        # Set LMC IP
        self.tpm.set_lmc_ip(self._lmc_ip, self._lmc_port)

        # Enable C2C streaming
        self.tpm["board.regfile.c2c_stream_enable"] = 0x0
        self.tpm["board.regfile.c2c_stream_enable"] = 0x1
        self.set_c2c_burst()

        # Synchronise FPGAs
        self.sync_fpga_time()

        # Initialise ADAs
        self.tpm.tpm_ada.initialise_adas()

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

    def program_fpgas(self, bitfile):
        """ Program FPGA with specified firmware
        :param bitfile: Bitfile to load """
        self.connect(simulation=True)
        self.logger.info("Downloading bitfile to board")
        if self.tpm is not None:
            self.tpm.download_firmware(Device.FPGA_1, bitfile)

    def program_cpld(self, bitfile):
        """ Program CPLD with specified bitfile
        :param bitfile: Bitfile to flash to CPLD"""
        self.connect(simulation=True)
        self.logger.info("Downloading bitstream to CPLD FLASH")
        if self.tpm is not None:
            return self.tpm.tpm_cpld.cpld_flash_write(bitfile)

    @connected
    def read_cpld(self, bitfile="cpld_dump.bit"):
        """
        Read bitfile in CPLD FLASH.

        :param bitfile: Bitfile where to dump CPLD firmware
        :type bitfile: str
        """
        self.logger.info("Reading bitstream from CPLD FLASH")
        self.tpm.tpm_cpld.cpld_flash_read(bitfile)

    @connected
    def print_fpga_firmware_information(self, fpga_id=0):
        """
        Print FPGA firmware information
        :param fpga_id: FPGA ID, 0 or 1
        :type fpga_id: int
        """
        if self.is_programmed():
            self.tpm.tpm_firmware_information[fpga_id].print_information()

    def get_ip(self):
        """
        Get tile IP.
        :return: tile IP address
        :rtype: str
        """
        return self._ip

    @connected
    def get_temperature(self):
        """
        Read board temperature.
        :return: board temperature
        :rtype: float
        """
        return self.tpm.temperature()

    @connected
    def get_voltage(self):
        """ Read board voltage """
        return self.tpm.voltage()

    @connected
    def get_current(self):
        """ Read board current """
        return self.tpm.current()

    @connected
    def get_rx_adc_rms(self):
        """
        Get ADC power.
        :return: ADC RMS power
        :rtype: list(float)
        """
        # If board is not programmed, return None
        if not self.tpm.is_programmed():
            return None

        # Get RMS values from board
        rms = []
        for adc_power_meter in self.tpm.adc_power_meter:
            rms.extend(adc_power_meter.get_RmsAmplitude())

        return rms

    @connected
    def get_adc_rms(self):
        """ Get ADC power
        :param adc_id: ADC ID"""

        # If board is not programmed, return None
        if not self.tpm.is_programmed():
            return None

        # Get RMS values from board
        rms = []
        for adc_power_meter in self.tpm.adc_power_meter:
            rms.extend(adc_power_meter.get_RmsAmplitude())

        # Re-map values
        return rms

    @connected
    def get_fpga0_temperature(self):
        """
        Get FPGA0 temperature
        :return: FPGA0 temperature
        :rtype: float
        """
        if self.is_programmed():
            return self.tpm.tpm_sysmon[0].get_fpga_temperature()
        else:
            return 0

    @connected
    def get_fpga1_temperature(self):
        """
        Get FPGA1 temperature
        :return: FPGA0 temperature
        :rtype: float
        """
        if self.is_programmed():
            return self.tpm.tpm_sysmon[1].get_fpga_temperature()
        else:
            return 0

    @connected
    def is_qsfp_module_plugged(self, qsfp_id=0):
        """
        Initialise firmware components.

        :return: True when cable is detected
        """
        qsfp_status = self['board.regfile.pll_10g']
        if qsfp_id == 0:
            qsfp_status = (qsfp_status >> 4) & 0x1
        else:
            qsfp_status = (qsfp_status >> 6) & 0x1
        if qsfp_status == 0:
            return True
        else:
            return False

    @connected
    def configure_10g_core(
        self,
        core_id,
        src_mac=None,
        src_ip=None,
        dst_mac=None,
        dst_ip=None,
        src_port=None,
        dst_port=None,
    ):
        """
        Configure a 10G core.

        :todo: Legacy method. Check whether to be deleted.

        :param core_id: 10G core ID
        :param src_mac: Source MAC address
        :param src_ip: Source IP address
        :param dst_mac: Destination MAC address
        :param dst_ip: Destination IP
        :param src_port: Source port
        :param dst_port: Destination port
        """
        # Configure core
        if src_mac is not None:
            self.tpm.tpm_10g_core[core_id].set_src_mac(src_mac)
        if src_ip is not None:
            self.tpm.tpm_10g_core[core_id].set_src_ip(src_ip)
        if dst_mac is not None:
            self.tpm.tpm_10g_core[core_id].set_dst_mac(dst_mac)
        if dst_ip is not None:
            self.tpm.tpm_10g_core[core_id].set_dst_ip(dst_ip)
        if src_port is not None:
            self.tpm.tpm_10g_core[core_id].set_src_port(src_port)
        if dst_port is not None:
            self.tpm.tpm_10g_core[core_id].set_dst_port(dst_port)

    @connected
    def configure_40g_core(
        self,
        core_id=0,
        arp_table_entry=0,
        src_mac=None,
        src_ip=None,
        src_port=None,
        dst_ip=None,
        dst_port=None,
        rx_port_filter=None,
        netmask=None,
        gateway_ip=None
    ):
        """
        Configure a 40G core.

        :param core_id: 40G core ID
        :param arp_table_entry: ARP table entry ID
        :param src_mac: Source MAC address
        :param src_ip: Source IP address
        :param dst_ip: Destination IP
        :param src_port: Source port
        :param dst_port: Destination port
        :param rx_port_filter: Filter for incoming packets
        :param netmask: Netmask
        :param gateway_ip: Gateway IP
        """
        # Configure core
        if src_mac is not None:
            self.tpm.tpm_10g_core[core_id].set_src_mac(src_mac)
        if src_ip is not None:
            self.tpm.tpm_10g_core[core_id].set_src_ip(src_ip)
        if dst_ip is not None:
            self.tpm.tpm_10g_core[core_id].set_dst_ip(dst_ip, arp_table_entry)
        if src_port is not None:
            self.tpm.tpm_10g_core[core_id].set_src_port(src_port, arp_table_entry)
        if dst_port is not None:
            self.tpm.tpm_10g_core[core_id].set_dst_port(dst_port, arp_table_entry)
        if rx_port_filter is not None:
            self.tpm.tpm_10g_core[core_id].set_rx_port_filter(
                rx_port_filter, arp_table_entry
            )
        if netmask is not None:
            self.tpm.tpm_10g_core[core_id].set_netmask(netmask)
        if gateway_ip is not None:
            self.tpm.tpm_10g_core[core_id].set_gateway_ip(gateway_ip)


    @connected
    def get_40g_core_configuration(self, core_id, arp_table_entry=0):
        """
        Get the configuration for a 40g core.

        :param core_id: Core ID
        :type core_id: int
        :param arp_table_entry: ARP table entry to use
        :type arp_table_entry: int

        :return: core configuration
        :rtype: dict
        """
        try:
            self._40g_configuration = {
                "core_id": core_id,
                "arp_table_entry": arp_table_entry,
                "src_mac": int(self.tpm.tpm_10g_core[core_id].get_src_mac()),
                "src_ip": int(self.tpm.tpm_10g_core[core_id].get_src_ip()),
                "dst_ip": int(
                    self.tpm.tpm_10g_core[core_id].get_dst_ip(arp_table_entry)
                ),
                "src_port": int(
                    self.tpm.tpm_10g_core[core_id].get_src_port(arp_table_entry)
                ),
                "dst_port": int(
                    self.tpm.tpm_10g_core[core_id].get_dst_port(arp_table_entry)
                ),
                "netmask": int(self.tpm.tpm_10g_core[core_id].get_netmask()),
                "gateway_ip": int(self.tpm.tpm_10g_core[core_id].get_gateway_ip()),
            }
        except IndexError:
            self._40g_configuration = None

        return self._40g_configuration

    @connected
    def set_default_eth_configuration(
            self,
            src_ip_fpga1=None,
            src_ip_fpga2=None,
            dst_ip_fpga1=None,
            dst_ip_fpga2=None,
            src_port=4661,
            dst_port=4660,
            qsfp_detection="auto"):
        """
        Set destination and source IP/MAC/ports for 40G cores.

        This will create a loopback between the two FPGAs.

        :param src_ip_fpga1: source IP address for FPGA1 40G interface
        :type src_ip_fpga1: str
        :param src_ip_fpga2: source IP address for FPGA2 40G interface
        :type src_ip_fpga2: str
        :param dst_ip_fpga1: destination IP address for beamformed data from FPGA1 40G interface
        :type dst_ip_fpga1: str
        :param dst_ip_fpga2: destination IP address for beamformed data from FPGA2 40G interface
        :type dst_ip_fpga2: str
        :param src_port: source UDP port for beamformed data packets
        :type src_port: int
        :param dst_port: destination UDP port for beamformed data packets
        :type dst_port: int

        :return: core configuration
        :rtype: dict

        """
        if self["fpga1.regfile.feature.xg_eth_implemented"] == 1:
            src_ip_list = [src_ip_fpga1, src_ip_fpga2]
            dst_ip_list = [dst_ip_fpga1, dst_ip_fpga2]

            for n in range(len(self.tpm.tpm_10g_core)):

                if qsfp_detection == "all":
                    cable_detected = True
                elif qsfp_detection == "auto" and self.is_qsfp_module_plugged(n):
                    cable_detected = True
                elif n == 0 and qsfp_detection == "qsfp1":
                    cable_detected = True
                elif n == 1 and qsfp_detection == "qsfp2":
                    cable_detected = True
                else:
                    cable_detected = False


                # generate src IP and MAC address
                if src_ip_list[n] is None:
                    src_ip_octets = self._ip.split(".")
                else:
                    src_ip_octets = src_ip_list[n].split(".")

                src_ip = f"10.0.{n + 1}.{src_ip_octets[3]}"
                dst_ip = dst_ip_list[n]

                # if QSFP cable is detected then reset core,
                # check for link up (done in reset reset_core) and set default IP address,
                # otherwise disable TX
                if cable_detected:
                    self.tpm.tpm_10g_core[n].reset_core()

                    self.configure_40g_core(
                        core_id=n,
                        arp_table_entry=0,
                        src_mac=0x620000000000 + ip2long(src_ip),
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        src_port=src_port,
                        dst_port=dst_port,
                        rx_port_filter=dst_port,
                    )
                else:
                    self.tpm.tpm_10g_core[n].tx_disable()

    @connected
    def set_lmc_download(
        self,
        mode,
        payload_length=1024,
        dst_ip=None,
        src_port=0xF0D0,
        dst_port=4660
    ):
        """
        Configure link and size of control data for LMC packets.

        :param mode: "1g" or "10g"
        :type mode: str
        :param payload_length: SPEAD payload length in bytes
        :type payload_length: int
        :param dst_ip: Destination IP
        :type dst_ip: str
        :param src_port: Source port for integrated data streams
        :type src_port: int
        :param dst_port: Destination port for integrated data streams
        :type dst_port: int
        """
        # Using 10G lane
        if mode.upper() == "10G":
            if payload_length >= 8193:
                self.logger.warning("Packet length too large for 10G")
                return

            # If dst_ip is None, use local lmc_ip
            if dst_ip is None:
                dst_ip = self._lmc_ip

            for core_id in range(len(self.tpm.tpm_10g_core)):
                self.configure_40g_core(
                    core_id=core_id,
                    arp_table_entry=1,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port
                )

            self["fpga1.lmc_gen.tx_demux"] = 2
            self["fpga2.lmc_gen.tx_demux"] = 2

        # Using dedicated 1G link
        elif mode.upper() == "1G":
            if dst_ip is not None:
                self._lmc_ip = dst_ip

            self.tpm.set_lmc_ip(self._lmc_ip, self._lmc_port)

            self["fpga1.lmc_gen.tx_demux"] = 1
            self["fpga2.lmc_gen.tx_demux"] = 1
        else:
            self.logger.warning("Supported modes are 1g, 10g")
            return

        self["fpga1.lmc_gen.payload_length"] = payload_length
        self["fpga2.lmc_gen.payload_length"] = payload_length

    @connected
    def set_lmc_integrated_download(
        self,
        mode,
        channel_payload_length,
        beam_payload_length,
        dst_ip=None,
        src_port=0xF0D0,
        dst_port=4660
    ):
        """
        Configure link and size of control data for integrated LMC packets.

        :param mode: '1g' or '10g'
        :type mode: str
        :param channel_payload_length: SPEAD payload length for integrated channel data
        :type channel_payload_length: int
        :param beam_payload_length: SPEAD payload length for integrated beam data
        :type beam_payload_length: int
        :param dst_ip: Destination IP
        :type dst_ip: str
        :param src_port: Source port for integrated data streams
        :type src_port: int
        :param dst_port: Destination port for integrated data streams
        :type dst_port: int
        """
        # Using 10G lane
        if mode.upper() == "10G":

            # If dst_ip is None, use local lmc_ip
            if dst_ip is None:
                dst_ip = self._lmc_ip

            for core_id in range(len(self.tpm.tpm_10g_core)):
                self.configure_40g_core(
                    core_id=core_id,
                    arp_table_entry=1,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port
                )

        # Using dedicated 1G link
        elif mode.upper() == "1G":
            pass
        else:
            self.logger.error("Supported mode are 1g, 10g")
            return

        # Setting payload lengths
        for i in range(len(self.tpm.tpm_integrator)):
            self.tpm.tpm_integrator[i].configure_download(
                mode, channel_payload_length, beam_payload_length
            )

    @connected
    def check_arp_table(self, timeout=20.0):
        """
        Check that ARP table has been resolved for all used cores.
        40G interfaces use cores 0 (fpga0) and 1 (fpga1) and
        ARP ID 0 for beamformer, 1 for LMC.
        The procedure checks that all populated ARP entries have been
        resolved. If the QSFP has been disabled or link is not detected up,
        the check is skipped.

        :param timeout: Timeout in seconds
        :type timeout: float
        :return: ARP table status
        :rtype: bool
        """

        # polling time to check ARP table
        polling_time = 0.1
        checks_per_second = 1.0 / polling_time
        # sanity check on time. Between 1 and 100 seconds
        max_time = int(timeout)
        if max_time < 1:
            max_time = 1
        if max_time > 100:
            max_time = 100
        # wait UDP link up
        core_id = range(len(self.tpm.tpm_10g_core))
        arp_table_id = range(self.tpm.tpm_10g_core[0].get_number_of_arp_table_entries())

        self.logger.info("Checking ARP table...")

        linked_core_id = []
        for c in core_id:
            if self.tpm.tpm_10g_core[c].is_tx_disabled():
                self.logger.warning("Skipping ARP table check on FPGA" + str(c+1) + ". TX is disabled!")
            elif self.tpm.tpm_10g_core[c].is_link_up():
                linked_core_id.append(c)
            else:
                self.logger.warning("Skipping ARP table check on FPGA" + str(c+1) + ". Link is down!")

        if not linked_core_id:
            return False

        times = 0
        while True:
            not_ready_links = []
            for c in linked_core_id:
                core_inst = self.tpm.tpm_10g_core[c]
                core_errors = core_inst.check_errors()
                if core_errors:
                    not_ready_links.append(c)
                for a in arp_table_id:
                    core_status, core_mac = core_inst.get_arp_table_status(a, silent_mode=True)
                    # check if valid entry has been resolved
                    if core_status & 0x1 == 1 and core_status & 0x4 == 0:
                        not_ready_links.append(c)

            if not not_ready_links:
                self.logger.info("40G Link established! ARP table populated!")
                return True
            else:
                times += 1
                time.sleep(polling_time)
                for c in linked_core_id:
                    if c in not_ready_links:
                        if times % checks_per_second == 0:
                            self.logger.warning(
                                f"40G Link on FPGA{c} not established after {int(0.1 * times)} seconds! Waiting... "
                            )
                        if times == max_time * checks_per_second:
                            self.logger.warning(
                                f"40G Link on FPGA{c} not established after {int(0.1 * times)} seconds! ARP table not populated!"
                            )
                            return False

    def get_arp_table(self):
        """
        Check that ARP table has been populated in for all used cores.
        Returns a dictionary with an entry for each core present in the firmware
        Each entry contains a list of the ARP table IDs which have been resolved
        by the ARP state machine.

        :return: list of populated core ids and arp table entries
        :rtype: dict(list)
        """
        # wait UDP link up
        if self["fpga1.regfile.feature.xg_eth_implemented"] == 1:
            self.logger.debug("Checking ARP table...")

            if self.tpm.tpm_debris_firmware[0].xg_40g_eth:
                core_ids = range(2)
                arp_table_ids = range(4)
            else:
                core_ids = range(8)
                arp_table_ids = [0]

            self._arp_table = {i: [] for i in core_ids}

            linkup = True
            for core_id in core_ids:
                for arp_table in arp_table_ids:
                    core_status, core_mac = self.tpm.tpm_10g_core[core_id].get_arp_table_status(
                        arp_table, silent_mode=True
                    )
                    if core_status & 0x4 == 0:
                        message = (
                            f"CoreID {core_id} with ArpID {arp_table} is not "
                            "populated"
                        )

                        self.logger.debug(message)
                        linkup = False
                    else:
                        self._arp_table[core_id].append(arp_table)

            if linkup:
                self.logger.debug("10G Link established! ARP table populated!")

        return self._arp_table

    @connected
    def set_station_id(self, station_id, tile_id):
        """
        Set station ID.

        :param station_id: Station ID
        :param tile_id: Tile ID within station
        """
        fpgas = ["fpga1", "fpga2"]
        if len(self.tpm.find_register("fpga1.regfile.station_id")) > 0:
            for f in fpgas:
                self[f + ".regfile.station_id"] = station_id
                self[f + ".regfile.tpm_id"] = tile_id
        else:
            for f in fpgas:
                self[f + ".dsp_regfile.config_id.station_id"] = station_id
                self[f + ".dsp_regfile.config_id.tpm_id"] = tile_id

    @connected
    def get_station_id(self):
        """
        Get station ID
        :return: station ID programmed in HW
        :rtype: int
        """
        if not self.tpm.is_programmed():
            return -1
        else:
            if len(self.tpm.find_register("fpga1.regfile.station_id")) > 0:
                tile_id = self["fpga1.regfile.station_id"]
            else:
                tile_id = self["fpga1.dsp_regfile.config_id.station_id"]
            return tile_id

    @connected
    def get_tile_id(self):
        """
        Get tile ID.

        :return: programmed tile id
        :rtype: int
        """
        if not self.tpm.is_programmed():
            return -1
        else:
            if len(self.tpm.find_register("fpga1.regfile.tpm_id")) > 0:
                tile_id = self["fpga1.regfile.tpm_id"]
            else:
                tile_id = self["fpga1.dsp_regfile.config_id.tpm_id"]
            return tile_id

    ###########################################
    # Time related methods
    ###########################################
    @connected
    def get_fpga_time(self, device):
        """
        Return time from FPGA.

        :param device: FPGA to get time from
        :type device: Device
        :return: Internal time for FPGA
        :rtype: int
        :raises LibraryError: Invalid value for device
        """
        if device == Device.FPGA_1:
            return self["fpga1.pps_manager.curr_time_read_val"]
        elif device == Device.FPGA_2:
            return self["fpga2.pps_manager.curr_time_read_val"]
        else:
            raise LibraryError("Invalid device specified")

    @connected
    def set_fpga_time(self, device, device_time):
        """
        Set Unix time in FPGA.

        :param device: FPGA to get time from
        :type device: Device
        :param device_time: Internal time for FPGA
        :type device_time: int
        :raises LibraryError: Invalid value for device
        """
        if device == Device.FPGA_1:
            self["fpga1.pps_manager.curr_time_write_val"] = device_time
            self["fpga1.pps_manager.curr_time_cmd.wr_req"] = 0x1
        elif device == Device.FPGA_2:
            self["fpga2.pps_manager.curr_time_write_val"] = device_time
            self["fpga2.pps_manager.curr_time_cmd.wr_req"] = 0x1
        else:
            raise LibraryError("Invalid device specified")

    @connected
    def get_fpga_timestamp(self, device=Device.FPGA_1):
        """
        Get timestamp from FPGA.

        :param device: FPGA to read timestamp from
        :type device: Device
        :return: PPS time
        :rtype: int
        :raises LibraryError: Invalid value for device
        """
        if device == Device.FPGA_1:
            return self["fpga1.pps_manager.timestamp_read_val"]
        elif device == Device.FPGA_2:
            return self["fpga2.pps_manager.timestamp_read_val"]
        else:
            raise LibraryError("Invalid device specified")

    @connected
    def get_phase_terminal_count(self):
        """
        Get PPS phase terminal count.

        :return: PPS phase terminal count
        :rtype: int
        """
        return self["fpga1.pps_manager.sync_tc.cnt_1_pulse"]

    @connected
    def set_phase_terminal_count(self, value):
        """
        Set PPS phase terminal count.

        :param value: PPS phase terminal count
        """
        self["fpga1.pps_manager.sync_tc.cnt_1_pulse"] = value
        self["fpga2.pps_manager.sync_tc.cnt_1_pulse"] = value

    @connected
    def get_pps_delay(self, enable_correction=False):
        """
        Get delay between PPS and 10 MHz clock.
        :param: enable_correction, enable PPS delay correction using value configured in the FPGA1
        :type: bool

        :return: delay between PPS and 10 MHz clock in 200 MHz cycles
        :rtype: int
        """
        if enable_correction:
            pps_correction = self["fpga1.pps_manager.sync_tc.cnt_2"]
            if pps_correction > 127:
                pps_correction -= 256
        else:
            pps_correction = 0
        return self["fpga1.pps_manager.sync_phase.cnt_hf_pps"] + pps_correction

    @connected
    def wait_pps_event(self):
        """
        Wait for a PPS edge. Added timeout feture to avoid method to stuck.

        :raises BoardError: Hardware PPS stuck
        """
        timeout = 1100
        t0 = self.get_fpga_time(Device.FPGA_1)
        while t0 == self.get_fpga_time(Device.FPGA_1):
            if timeout > 0:
                time.sleep(0.001)
                timeout = timeout - 1
                pass
            else:
                raise BoardError("TPM PPS counter does not advance")

    @connected
    def wait_pps_event2(self):
        """ Wait for a PPS edge """
        self['fpga1.pps_manager.pps_edge.req'] = 1
        while self['fpga1.pps_manager.pps_edge.req'] == 1:
            time.sleep(0.01)

    @connected
    def check_pending_data_requests(self):
        """
        Checks whether there are any pending data requests.

        :return: true if pending requests are present
        :rtype: bool
        """
        return (self["fpga1.lmc_gen.request"] + self["fpga2.lmc_gen.request"]) > 0

    ########################################################
    # channeliser
    ########################################################
    @connected
    def download_polyfilter_coeffs(self, window="hann", bin_width_scaling=1.0):
        for n in range(2):
            self.tpm.polyfilter[n].set_window(window, bin_width_scaling)
            self.tpm.polyfilter[n].download_coeffs()

    #######################################################################################

    @connected
    def configure_channeliser(self):
        self.logger.info("Configuring channeliser...")
        return
        # self['fpga1.dsp_regfile.adc_remap.enable'] = 1
        # self['fpga2.dsp_regfile.adc_remap.enable'] = 1
        # self['fpga1.dsp_regfile.adc_remap.lsb_discard'] = 6
        # self['fpga2.dsp_regfile.adc_remap.lsb_discard'] = 6

    @connected
    def set_channeliser_truncation(self, trunc):
        """ Set channeliser truncation scale """
        self['fpga1.dsp_regfile.channelizer_out_bit_round'] = trunc
        self['fpga2.dsp_regfile.channelizer_out_bit_round'] = trunc
        # trunc16 = 4
        # self['fpga1.dsp_regfile.channelizer_out_bit_round16'] = trunc16
        # self['fpga2.dsp_regfile.channelizer_out_bit_round16'] = trunc16
        return

    def set_fft_shift(self, shift):
        try:
            self['fpga1.dsp_regfile.channelizer_fft_shift'] = shift
            self['fpga2.dsp_regfile.channelizer_fft_shift'] = shift
        except:
            pass

    # ---------------------------- Synchronisation routines ------------------------------------
    @connected
    def post_synchronisation(self):
        """ Post tile configuration synchronization """

        self.wait_pps_event()

        current_tc = self.get_phase_terminal_count()
        delay = self.get_pps_delay()

        self.set_phase_terminal_count(self.calculate_delay(delay, current_tc, 20, 4))

        self.wait_pps_event()

        delay = self.get_pps_delay()
        self.logger.info("Finished tile post synchronisation ({})".format(delay))

    # ------------------------------------
    # Synchronisation routines
    # ------------------------------------
    @connected
    def sync_fpga_time(self, use_internal_pps=False):
        """Set UTC time to two FPGAs in the tile Returns when these are synchronised.

        :param use_internal_pps: use internally generated PPS, for test/debug
        :type use_internal_pps: bool
        """

        devices = ["fpga1", "fpga2"]

        # Setting internal PPS generator
        for f in devices:
            self.tpm[f + ".pps_manager.pps_gen_tc"] = int(100e6) - 1  # PPS generator runs at 100 Mhz
            self.tpm[f + ".pps_manager.sync_cnt_enable"] = 0x7
            self.tpm[f + ".pps_manager.sync_cnt_enable"] = 0x0
            # if self.tpm.has_register("fpga1.pps_manager.pps_exp_tc"):
            #     self.tpm[f + ".pps_manager.pps_exp_tc"] = int(200e6) - 1  # PPS validation runs at 200 Mhz
            # else:
            #     self.logger.info("FPGA Firmware does not support updated PPS validation. Status of PPS error flag should be ignored.")

        # Setting internal PPS generator
        if use_internal_pps:
            for f in devices:
                self.tpm[f + ".regfile.spi_sync_function"] = 1
                self.tpm[f + ".pps_manager.pps_gen_sync"] = 0
                self.tpm[f + ".pps_manager.pps_gen_sync.enable"] = 1
            time.sleep(0.1)
            self.tpm["fpga1.pps_manager.pps_gen_sync.act"] = 1
            time.sleep(0.1)
            for f in devices:
                self.tpm[f + ".pps_manager.pps_gen_sync"] = 0
                self.tpm[f + ".regfile.spi_sync_function"] = 1
                self.tpm[f + ".pps_manager.pps_selection"] = 1
            self.logger.warning("Using Internal PPS generator!")
            self.logger.info("Internal PPS generator synchronised.")

        # Setting UTC time
        max_attempts = 5
        for _n in range(max_attempts):
            self.logger.info("Synchronising FPGA UTC time.")
            self.wait_pps_event2()
            time.sleep(0.5)

            t = int(time.time())
            self.set_fpga_time(Device.FPGA_1, t)
            self.set_fpga_time(Device.FPGA_2, t)

            # configure the PPS sampler
            self.set_pps_sampling(20, 4)

            self.wait_pps_event2()
            time.sleep(0.1)
            t0 = self.tpm["fpga1.pps_manager.curr_time_read_val"]
            t1 = self.tpm["fpga2.pps_manager.curr_time_read_val"]

            if t0 == t1:
                return

        self.logger.error("Not possible to synchronise FPGA UTC time after " + str(max_attempts) + " attempts!")

    @connected
    def set_pps_sampling(self, target, margin):
        """
        Set the PPS sampler terminal count

        :param target: target delay
        :type target: int
        :param margin: margin, target +- margin
        :type margin: int
        """

        current_tc = self.get_phase_terminal_count()
        current_delay = self.get_pps_delay()
        self.set_phase_terminal_count(self.calculate_delay(current_delay,
                                                           current_tc,
                                                           target,
                                                           margin))


    @connected
    def check_server_time(self):
        self.wait_pps_event()
        fpga_time = self.tpm["fpga1.pps_manager.curr_time_read_val"]
        server_time = datetime.datetime.now()
        print("Server Time: " + str(server_time.timestamp()))
        print("FPGA Time: " + str(fpga_time))

    @connected
    def check_fpga_synchronization(self):
        """
        Checks various synchronization parameters.

        Output in the log

        :return: OK status
        :rtype: bool
        """
        result = True
        # check PLL status
        pll_status = self.tpm["pll", 0x508]
        if pll_status == 0xE7:
            self.logger.debug("PLL locked to external reference clock.")
        elif pll_status == 0xF2:
            self.logger.warning("PLL locked to internal reference clock.")
        else:
            self.logger.error(
                "PLL is not locked! - Status Readback 0 (0x508): " + hex(pll_status)
            )
            result = False

        # check PPS detection
        if self.tpm["fpga1.pps_manager.pps_detected"] == 0x1:
            self.logger.debug("FPGA1 is locked to external PPS")
        else:
            self.logger.warning("FPGA1 is not locked to external PPS")
        if self.tpm["fpga2.pps_manager.pps_detected"] == 0x1:
            self.logger.debug("FPGA2 is locked to external PPS")
        else:
            self.logger.warning("FPGA2 is not locked to external PPS")
        
        # Check PPS valid
        if self.tpm.has_register("fpga1.pps_manager.pps_exp_tc"):
            if self.tpm[f'fpga1.pps_manager.pps_errors.pps_count_error'] == 0x0:
                self.logger.debug("FPGA1 PPS period is as expected.")
            else:
                self.logger.error("FPGA1 PPS period is not as expected.")
                result = False
        else:
            self.logger.info("FPGA1 Firmware does not support updated PPS validation. Ignoring status of PPS error flag.")
        if self.tpm.has_register("fpga2.pps_manager.pps_exp_tc"):
            if self.tpm[f'fpga2.pps_manager.pps_errors.pps_count_error'] == 0x0:
                self.logger.debug("FPGA2 PPS period is as expected.")
            else:
                self.logger.error("FPGA2 PPS period is not as expected.")
                result = False
        else:
            self.logger.info("FPGA2 Firmware does not support updated PPS validation. Ignoring status of PPS error flag.")  

        # check FPGA time
        self.wait_pps_event()
        t0 = self.tpm["fpga1.pps_manager.curr_time_read_val"]
        t1 = self.tpm["fpga2.pps_manager.curr_time_read_val"]
        self.logger.info("FPGA1 time is " + str(t0))
        self.logger.info("FPGA2 time is " + str(t1))
        if t0 != t1:
            self.logger.error("Time different between FPGAs detected!")
            result = False

        # check FPGA timestamp
        t0 = self.tpm["fpga1.pps_manager.timestamp_read_val"]
        t1 = self.tpm["fpga2.pps_manager.timestamp_read_val"]
        self.logger.info("FPGA1 timestamp is " + str(t0))
        self.logger.info("FPGA2 timestamp is " + str(t1))
        if abs(t0 - t1) > 1:
            self.logger.warning("Timestamp different between FPGAs detected!")

        # Check FPGA ring beamfomrer timestamp
        # t0 = self.tpm["fpga1.beamf_ring.current_frame"]
        # t1 = self.tpm["fpga2.beamf_ring.current_frame"]
        # self.logger.info("FPGA1 station beamformer timestamp is " + str(t0))
        # self.logger.info("FPGA2 station beamformer timestamp is " + str(t1))
        # if abs(t0 - t1) > 1:
        #     self.logger.warning(
        #         "Beamformer timestamp different between FPGAs detected!"
        #     )

        return result

    @connected
    def set_c2c_burst(self):
        """Setting C2C burst when supported by FPGAs and CPLD."""
        self.tpm["fpga1.regfile.c2c_stream_ctrl.idle_val"] = 0
        self.tpm["fpga2.regfile.c2c_stream_ctrl.idle_val"] = 0
        if len(self.tpm.find_register("fpga1.regfile.feature.c2c_linear_burst")) > 0:
            fpga_burst_supported = self.tpm["fpga1.regfile.feature.c2c_linear_burst"]
        else:
            fpga_burst_supported = 0
        if len(self.tpm.find_register("board.regfile.c2c_ctrl.mm_burst_enable")) > 0:
            self.tpm["board.regfile.c2c_ctrl.mm_burst_enable"] = 0
            cpld_burst_supported = 1
        else:
            cpld_burst_supported = 0

        if cpld_burst_supported == 1 and fpga_burst_supported == 1:
            self.tpm["board.regfile.c2c_ctrl.mm_burst_enable"] = 1
            self.logger.debug("C2C burst activated.")
            return
        if fpga_burst_supported == 0:
            self.logger.debug("C2C burst is not supported by FPGAs.")
        if cpld_burst_supported == 0:
            self.logger.debug("C2C burst is not supported by CPLD.")

    @connected
    def synchronised_data_operation(self, seconds=0.2, timestamp=None):
        """
        Synchronise data operations between FPGAs.

        :param seconds: Number of seconds to delay operation
        :param timestamp: Timestamp at which tile will be synchronised

        :return: timestamp written into FPGA timestamp request register
        :rtype: int
        """
        # Wait while previous data requests are processed
        while (
            self.tpm["fpga1.lmc_gen.request"] != 0
            or self.tpm["fpga2.lmc_gen.request"] != 0
        ):
            self.logger.info("Waiting for data request to be cleared by firmware...")
            time.sleep(0.05)

        self.logger.debug("Command accepted")

        # Read timestamp
        if timestamp is not None:
            t0 = timestamp
        else:
            t0 = max(
                self.tpm["fpga1.pps_manager.timestamp_read_val"],
                self.tpm["fpga2.pps_manager.timestamp_read_val"],
            )

        # Set arm timestamp
        # delay = number of frames to delay * frame time (shift by 8)
        frame_time = 1.0 / (700e6 / 8.0) * 1024
        delay = seconds * (1 / frame_time) / 256
        t1 = t0 + int(delay)
        for fpga in self.tpm.tpm_fpga:
            fpga.fpga_apply_sync_delay(t1)
        return t1


    @connected
    def check_valid_timestamp_request(
        self, daq_mode, fpga_id=None
    ):
        """
        Check valid timestamp request for various modes
        modes supported: raw_adc, channelizer and beamformer

        :param daq_mode: string used to select which Flag register of the LMC to read
        :param fpga_id: FPGA_ID, 0 or 1. Default None will select both FPGAs

        :return: boolean to indicate if the timestamp request is valid or not
        :rtype: boolean
        """
        C_VALID_TIMESTAMP_REQ = 0
        list_of_valid_timestamps = []
        fpga_list = range(len(self.tpm.tpm_debris_firmware)) if fpga_id is None else [fpga_id]

        if daq_mode not in self.daq_modes_with_timestamp_flag:
            raise LibraryError(f"Invalid daq_mode specified: {daq_mode} not supported")

        for fpga in fpga_list:
            valid_request = self[f"fpga{fpga + 1}.lmc_gen.timestamp_req_invalid.{daq_mode}"] == C_VALID_TIMESTAMP_REQ
            list_of_valid_timestamps.append(valid_request)
            self.logger.debug(f"fpga{fpga + 1} {daq_mode} timestamp request is: {'VALID' if valid_request else 'INVALID'}")
        if not all(list_of_valid_timestamps):
            self.logger.error("INVALID LMC Data request")
            return False
        else:
            return True

    def select_method_to_check_valid_synchronised_data_request(
            self, daq_mode, t_request, fpga_id=None
    ):
        """
        Checks if Firmware contains the invalid flag register that raises a flag during synchronisation error.
        If the Firmware has the register then it will read it to check that the timestamp request was valid.
        If the register is not present, the software method will be used to calculate if the timestamp request was valid

        :param daq_mode: string used to select which Flag register of the LMC to read
        :param t_request: requested timestamp. Must be more than current timestamp to be synchronised successfuly
        :param fpga_id: FPGA_ID, 0 or 1. Default None
        """
        timestamp_invalid_flag_supported = self.tpm.has_register(f"fpga1.lmc_gen.timestamp_req_invalid.{daq_mode}")
        if timestamp_invalid_flag_supported:
            valid_request = self.check_valid_timestamp_request(daq_mode, fpga_id)
        else:
            self.logger.warning(
                "FPGA firmware doesn't support invalid data request flag, request will be validated by software"
            )
            valid_request = self.check_synchronised_data_operation(t_request)
        if valid_request:
            self.logger.info(f"Valid {daq_mode} Timestamp request")
            return
        self.clear_lmc_data_request()
        if timestamp_invalid_flag_supported:
            self.clear_timestamp_invalid_flag_register(daq_mode, fpga_id)
        self.logger.info("LMC Data request has been cleared")
        return

    @connected
    def check_synchronised_data_operation(self, requested_timestamp=None):
        """
        Check if synchronise data operations between FPGAs is successful.

        :param requested_timestamp: Timestamp written into FPGA timestamp request register, if None it will be read
        from the FPGA register

        :return: Operation success
        :rtype: bool
        """
        if requested_timestamp is None:
            t_arm1 = self.tpm["fpga1.pps_manager.timestamp_req_val"]
            t_arm2 = self.tpm["fpga2.pps_manager.timestamp_req_val"]
        else:
            t_arm1 = requested_timestamp
            t_arm2 = requested_timestamp
        t_now1 = self.tpm["fpga1.pps_manager.timestamp_read_val"]
        t_now2 = self.tpm["fpga2.pps_manager.timestamp_read_val"]
        t_now_max = max(t_now1, t_now2)
        t_arm_min = min(t_arm1, t_arm2)
        t_margin = t_arm_min - t_now_max
        if t_margin <= 0:
            self.logger.error("Synchronised operation failed!")
            self.logger.error("Requested timestamp: " + str(t_arm_min))
            self.logger.error("Current timestamp: " + str(t_now_max))
            return False
        self.logger.debug("Synchronised operation successful!")
        self.logger.debug("Requested timestamp: " + str(t_arm_min))
        self.logger.debug("Current timestamp: " + str(t_now_max))
        self.logger.debug("Margin: " + str((t_arm_min - t_now_max) * 256 * 1.08e-6) + "s")
        return True

    @connected
    def configure_integrated_channel_data(self, integration_time=0.5):
        """ Configure continuous integrated channel data """
        for i in range(len(self.tpm.tpm_integrator)):
            self.tpm.tpm_integrator[i].configure("channel", integration_time, first_channel=0, last_channel=2048,
                                                 time_mux_factor=1, carousel_enable=0x0, download_bit_width=32,
                                                 data_bit_width=12)

    @connected
    def stop_integrated_channel_data(self):
        """ Stop transmission of integrated beam data"""
        for i in range(len(self.tpm.tpm_integrator)):
            self.tpm.tpm_integrator[i].stop_integrated_channel_data()

    @connected
    def stop_integrated_data(self):
        """ Stop transmission of integrated data"""
        for i in range(len(self.tpm.tpm_integrator)):
            self.tpm.tpm_integrator[i].stop_integrated_channel_data()

    @connected
    def start_acquisition(self, start_time=None, delay=2):
        """
        Start data acquisition.

        :param start_time: Time for starting (frames)
        :param delay: delay after start_time (frames)
        """
        devices = ["fpga1", "fpga2"]
        for f in devices:
            self.tpm[f + ".regfile.eth10g_ctrl"] = 0x0

        # Temporary (moved here from TPM control)
        if len(self.tpm.find_register("fpga1.regfile.c2c_stream_header_insert")) > 0:
            self.tpm["fpga1.regfile.c2c_stream_header_insert"] = 0x1
            self.tpm["fpga2.regfile.c2c_stream_header_insert"] = 0x1
        else:
            self.tpm["fpga1.regfile.c2c_stream_ctrl.header_insert"] = 0x1
            self.tpm["fpga2.regfile.c2c_stream_ctrl.header_insert"] = 0x1

        if len(self.tpm.find_register("fpga1.regfile.lmc_stream_demux")) > 0:
            self.tpm["fpga1.regfile.lmc_stream_demux"] = 0x1
            self.tpm["fpga2.regfile.lmc_stream_demux"] = 0x1

        for f in devices:
            # Disable start force (not synchronised start)
            self.tpm[f + ".pps_manager.start_time_force"] = 0x0
            self.tpm[f + ".lmc_gen.timestamp_force"] = 0x0

        # Read current sync time
        if start_time is None:
            t0 = self.tpm["fpga1.pps_manager.curr_time_read_val"]
        else:
            t0 = start_time

        sync_time = t0 + delay
        # Write start time
        for f in devices:
            self.tpm[f + ".pps_manager.sync_time_val"] = sync_time

    @staticmethod
    def calculate_delay(current_delay, current_tc, target, margin):
        """
        Calculate delay for PPS pulse.

        :param current_delay: Current delay
        :type current_delay: int
        :param current_tc: Current phase register terminal count
        :type current_tc: int
        :param target: target delay
        :type target: int
        :param margin: marging, target +-margin
        :type margin: int
        :return: Modified phase register terminal count
        :rtype: int
        """
        ref_low = target - margin
        ref_hi = target + margin
        for n in range(5):
            if current_delay <= ref_low:
                new_delay = current_delay + int((n * 40) / 5)
                new_tc = (current_tc + n) % 5
                if new_delay >= ref_low:
                    return new_tc
            elif current_delay >= ref_hi:
                new_delay = current_delay - int((n * 40) / 5)
                new_tc = current_tc - n
                if new_tc < 0:
                    new_tc += 5
                if new_delay <= ref_hi:
                    return new_tc
            else:
                return current_tc

    # ------------------------------------
    # Wrapper for data acquisition: RAW
    # ------------------------------------
    @connected
    def send_raw_data(
        self, sync=False, timestamp=None, seconds=0.2, fpga_id=None
    ):
        """ Send raw data from the TPM
        :param sync: Synchronised flag
        :param timestamp: When to start
        :param seconds: Delay
        :param fpga_id: Specify which FPGA should transmit, 0,1, or None for both FPGAs"""

        self.stop_data_transmission()
        # Data transmission should be synchronised across FPGAs
        t_request = self.synchronised_data_operation(timestamp=timestamp, seconds=seconds)

        # Send data from all FPGAs
        if fpga_id is None:
            fpgas = range(len(self.tpm.tpm_debris_firmware))
        else:
            fpgas = [fpga_id]
        for i in fpgas:
            if sync:
                self.tpm.tpm_debris_firmware[i].send_raw_data_synchronised()
            else:
                self.tpm.tpm_debris_firmware[i].send_raw_data()

        # Check if synchronisation is successful
        self.select_method_to_check_valid_synchronised_data_request("raw_adc_mode", t_request, fpga_id)

    @connected
    def send_raw_data_synchronised(
        self, timestamp=None, seconds=0.2
    ):
        """  Send synchronised raw data
        :param timestamp: When to start
        :param seconds: Period"""
        self.send_raw_data(
            sync=True,
            timestamp=timestamp,
            seconds=seconds,
        )

    @connected
    def stop_data_transmission(self):
        """ Stop all data transmission from TPM"""
        # self.logger.info("Stopping all transmission")
        # # All data format transmission except channelised data continuous stops autonomously
        # self.stop_channelised_data_continuous()
        return

    # ---------------------------- Wrapper for test generator ----------------------------

    def test_generator_set_tone(self, dds, frequency=100e6, ampl=0.0, phase=0.0, delay=128):
        """ Set test generator tone """
        translated_frequency = frequency - self._ddc_frequency + self._sampling_rate / (self._decimation_ratio * 4.0)
        self.logger.info("DDC Frequency: {}, translated frequency: {}".format(self._ddc_frequency, translated_frequency))

        t0 = self.tpm["fpga1.pps_manager.timestamp_read_val"]
        self.tpm.test_generator[0].set_tone(dds, translated_frequency, ampl, phase, t0 + delay)
        self.tpm.test_generator[1].set_tone(dds, translated_frequency, ampl, phase, t0 + delay)
        t1 = self.tpm["fpga1.pps_manager.timestamp_read_val"]
        if t1 >= t0 + delay or t1 <= t0:
            self.logger.info("Set tone test pattern generators synchronisation failed.")
            self.logger.info("Start Time   = " + str(t0))
            self.logger.info("Finish time  = " + str(t1))
            self.logger.info("Maximum time = " + str(t0 + delay))
            return -1
        return 0

    def test_generator_disable_tone(self, dds, delay=128):
        t0 = self.tpm["fpga1.pps_manager.timestamp_read_val"]
        self.tpm.test_generator[0].set_tone(dds, 0, 0, 0, t0 + delay)
        self.tpm.test_generator[1].set_tone(dds, 0, 0, 0, t0 + delay)
        t1 = self.tpm["fpga1.pps_manager.timestamp_read_val"]
        if t1 >= t0 + delay or t1 <= t0:
            self.logger.info("Set tone test pattern generators synchronisation failed.")
            self.logger.info("Start Time   = " + str(t0))
            self.logger.info("Finish time  = " + str(t1))
            self.logger.info("Maximum time = " + str(t0 + delay))
            return -1
        return 0

    def test_generator_set_noise(self, ampl=0.0, delay=128):
        t0 = self.tpm["fpga1.pps_manager.timestamp_read_val"]
        self.tpm.test_generator[0].enable_prdg(ampl, t0 + delay)
        self.tpm.test_generator[1].enable_prdg(ampl, t0 + delay)
        t1 = self.tpm["fpga1.pps_manager.timestamp_read_val"]
        if t1 >= t0 + delay or t1 <= t0:
            self.logger.info("Set tone test pattern generators synchronisation failed.")
            self.logger.info("Start Time   = " + str(t0))
            self.logger.info("Finish time  = " + str(t1))
            self.logger.info("Maximum time = " + str(t0 + delay))
            return -1
        return 0

    def test_generator_input_select(self, inputs):
        self.tpm.test_generator[0].channel_select(inputs & 0xFFFF)
        self.tpm.test_generator[1].channel_select((inputs >> 16) & 0xFFFF)

    # ---------------------------- Polyphase configuration ----------------------------

    def load_default_poly_coeffs(self):
        """ Load Channeliser coefficients """
        N = self['fpga1.poly.config1.length']
        S = self['fpga1.poly.config1.stages']
        MUX = self['fpga1.poly.config1.mux']
        C = self['fpga1.poly.config2.coeff_data_width']
        MUX_PER_RAM = self['fpga1.poly.config2.coeff_mux_per_ram']
        NOF_RAM_PER_STAGE = MUX / MUX_PER_RAM
        M = N * S

        base_width = C
        while base_width > 32:
            base_width /= 2
        aspect_ratio_coeff = C / base_width

        coeff = np.zeros(M, dtype=int)
        for i in range(M):
            real_val = np.sinc((float(i) - float(M / 2)) / float(N))  # sinc
            real_val *= 0.5 - 0.5 * np.cos(2 * np.pi * float(i) / float(M))  # window
            real_val *= 2 ** (C - 1) - 1  # rescaling
            coeff[i] = int(real_val)

        coeff_ram = np.zeros(N / NOF_RAM_PER_STAGE, dtype=int)
        for s in range(S):
            for ram in range(NOF_RAM_PER_STAGE):
                idx = 0
                for n in range(N):
                    if (n % MUX) / MUX_PER_RAM == ram:
                        coeff_ram[idx] = coeff[N * s + n]
                        idx += 1

                if aspect_ratio_coeff > 1:
                    coeff_ram_arc = np.zeros(N / NOF_RAM_PER_STAGE * aspect_ratio_coeff, dtype=int)
                    for n in range(N / NOF_RAM_PER_STAGE):
                        for m in range(aspect_ratio_coeff):
                            coeff_ram_arc[n * aspect_ratio_coeff + m] = coeff_ram[n] >> (m * C / aspect_ratio_coeff)
                else:
                    coeff_ram_arc = coeff_ram

                self['fpga1.poly.address.mux_ptr'] = ram
                self['fpga1.poly.address.stage_ptr'] = s
                self['fpga2.poly.address.mux_ptr'] = ram
                self['fpga2.poly.address.stage_ptr'] = s
                self['fpga1.poly.coeff'] = coeff_ram_arc.tolist()
                self['fpga2.poly.coeff'] = coeff_ram_arc.tolist()

    def set_fpga_sysref_gen(self, sysref_period):
        self['fpga1.pps_manager.sysref_gen_period'] = sysref_period - 1
        self['fpga1.pps_manager.sysref_gen_duty'] = sysref_period // 2 - 1
        self['fpga1.pps_manager.sysref_gen.enable'] = 1
        self['fpga1.pps_manager.sysref_gen.spi_sync_enable'] = 1
        self['fpga1.pps_manager.sysref_gen.sysref_pol_invert'] = 0
        self['fpga1.regfile.sysref_fpga_out_enable'] = 1

    def write_adc_broadcast(self, add, data, wait_sync=0):
        cmd = 1 + 0x8 * wait_sync
        self['board.spi'] = [add, data << 8, 0, 0xF, 0xF, cmd]

    def __str__(self):
        return str(self.tpm)

    def __getitem__(self, key):
        return self.tpm[key]

    def __setitem__(self, key, value):
        self.tpm[key] = value

    def __getattr__(self, name):
        if name in dir(self.tpm):
            return getattr(self.tpm, name)
        else:
            raise AttributeError("'Tile' or 'TPM' object have no attribute {}".format(name))

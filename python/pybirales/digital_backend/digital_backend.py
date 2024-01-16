#! /usr/bin/env python
#from pymexart.digital_backend.tile_mexart import Tile
from pybirales.digital_backend.tile_debris import Tile
from pyfabil import Device

from multiprocessing import Pool
from threading import Thread
import threading
import logging
import yaml
import time
import math
import sys
import os

# Define default configuration
configuration = {'tiles': None,
                 'station': {
                     'id': 0,
                     'name': "debris",
                     'program': False,
                     'initialise': False,
                     'program_cpld': False,
                     'enable_test': False,
                     'bitfile': None,
                     'channel_truncation': 7,
                     'channel_integration_time': -1,
                     'ada_gain': None
                     },
                 'observation': {
                     'sampling_frequency': 700e6,
                     'bandwidth': 12.5e6,
                     'ddc_frequency': 139.65e6
                     },
                 'network': {
                     'lmc': {
                         'tpm_cpld_port': 10000,
                         'lmc_ip': "10.0.10.200",
                         'use_teng': True,
                         'lmc_port': 4660,
                         'lmc_mac': 0x248A078F9D38,
                         'integrated_data_ip': "10.0.0.2",
                         'integrated_data_port': 5000,
                         'use_teng_integrated': True}
                    }
                 }


def create_tile_instance(config, tile_number):
    """ Add a new tile to the station
    :param config: Station configuration
    :param tile_ip: TPM to associate tile to """

    # If all traffic is going through 1G then set the destination port to
    # the lmc_port. If only integrated data is going through the 1G set the
    # destination port to integrated_data_port
    dst_port = config['network']['lmc']['lmc_port']
    lmc_ip = config['network']['lmc']['lmc_ip']

    if not config['network']['lmc']['use_teng_integrated'] and \
            config['network']['lmc']['use_teng']:
        dst_port = config['network']['lmc']['integrated_data_port']
        lmc_ip = config['network']['lmc']['integrated_data_ip']
    ddc_frequency = config['observation']['ddc_frequency']
    sampling_frequency = config['observation']['sampling_frequency']

    return Tile(config['tiles'][tile_number],
                config['network']['lmc']['tpm_cpld_port'],
                lmc_ip,
                dst_port,
                sampling_rate=sampling_frequency,
                ddc_frequency=ddc_frequency)


def program_cpld(params):
    """ Update tile CPLD.
     :param params: Contain 0) Station configuration and 1) Tile number to program CPLD """
    config, tile_number = params

    try:
        threading.current_thread().name = config['tiles'][tile_number]
        logging.info("Initialising Tile {}".format(config['tiles'][tile_number]))

        # Create station instance and program CPLD
        station_tile = create_tile_instance(config, tile_number)
        station_tile.program_cpld(config['station']['bitfile'])
        return True
    except Exception as e:
        logging.error("Could not program CPLD of {}: {}".format(config['tiles'][tile_number], e))
        return False


def program_fpgas(params):
    """ Program FPGAs
     :param params: Contain 0) Station configuration and 1) Tile number to program FPGAs """
    config, tile_number = params

    try:
        threading.current_thread().name = config['tiles'][tile_number]
        logging.info("Initialising Tile {}".format(config['tiles'][tile_number]))

        # Create station instance and program FPGAs
        station_tile = create_tile_instance(config, tile_number)
        station_tile.program_fpgas(config['station']['bitfile'])
        return True
    except Exception as e:
        logging.error("Could not program FPGAs of {}: {}".format(config['tiles'][tile_number], e))
        return False


def initialise_tile(params):
    """ Internal connect method to thread connection
     :param params: Contain 0) Station configuration and 1) Tile number to initialise """
    config, tile_number = params

    try:
        threading.current_thread().name = config['tiles'][tile_number]
        logging.info("Initialising Tile {}".format(config['tiles'][tile_number]))
        threading.current_thread().name = config['tiles'][tile_number]

        # Create station instance and initialise
        station_tile = create_tile_instance(config, tile_number)
        station_tile.initialise(enable_ada=True, enable_test=config['station']['enable_test'])

        # Set channeliser truncation
        station_tile.set_channeliser_truncation(config['station']['channel_truncation'])

        # Configure channel and beam integrated data
        station_tile.stop_integrated_data()
        if config['station']['channel_integration_time'] != -1:
            station_tile.configure_integrated_channel_data(
                config['station']['channel_integration_time'])

        # Enable SPEAD transmission
        station_tile['fpga1.dsp_regfile.spead_tx_enable'] = 1
        station_tile['fpga2.dsp_regfile.spead_tx_enable'] = 1

        # Configure polyphase filterbank and set 1G stream
        station_tile.download_polyfilter_coeffs("hann")
        station_tile.set_lmc_download("1g")
        station_tile.set_lmc_integrated_download("1g")
        station_tile['board.regfile.ethernet_pause'] = 0x0400

        return True
    except Exception as e:
        logging.warning("Could not initialise Tile {}: {}".format(config['tiles'][tile_number], e))
        return False


class Station(object):
    """ Class representing an AAVS station """

    def __init__(self, config):
        """ Class constructor
         :param config: Configuration dictionary for station """

        # Save configuration locally
        self.configuration = config
        self._station_id = config['station']['id']

        # Add tiles to station
        self.tiles = []
        for tile in config['tiles']:
            self.add_tile(tile)

        # Default duration of sleeps
        self._seconds = 0.5

        # Set if the station is properly configured
        self.properly_formed_station = None

    def add_tile(self, tile_ip):
        """ Add a new tile to the station
        :param tile_ip: Tile IP to be added to station """

        # If all traffic is going through 1G then set the destination port to
        # the lmc_port. If only integrated data is going through the 1G set the
        # destination port to integrated_data_port
        dst_port = self.configuration['network']['lmc']['lmc_port']
        lmc_ip = self.configuration['network']['lmc']['lmc_ip']
        
        if not self.configuration['network']['lmc']['use_teng_integrated'] and \
            self.configuration['network']['lmc']['use_teng']:
            dst_port = self.configuration['network']['lmc']['integrated_data_port']
            lmc_ip = self.configuration['network']['lmc']['integrated_data_ip']

        ddc_frequency = self.configuration['observation']['ddc_frequency']
        sampling_frequency = self.configuration['observation']['sampling_frequency']

        self.tiles.append(Tile(tile_ip,
                               self.configuration['network']['lmc']['tpm_cpld_port'],
                               lmc_ip,
                               dst_port,
                               sampling_rate=sampling_frequency,
                               ddc_frequency=ddc_frequency))

    def connect(self):
        """ Initialise all tiles """

        # Start with the assumption that the station will be properly formed
        self.properly_formed_station = True

        # Create a pool of nof_tiles processes
        pool = None
        if any([self.configuration['station']['program_cpld'], 
                self.configuration['station']['program'], 
                self.configuration['station']['initialise']]):
            pool = Pool(len(self.tiles))

        # Create parameters for processes
        params = tuple([(self.configuration, i) for i in range(len(self.tiles))])

        # Check if we are programming the CPLD, and if so program
        if self.configuration['station']['program_cpld']:
            logging.info("Programming CPLD")
            res = pool.map(program_cpld, params)

            if not all(res):
                logging.error("Could not program TPM CPLD!")
                self.properly_formed_station = False

        # Check if programming is required, and if so program
        if self.configuration['station']['program'] and self.properly_formed_station:
            logging.info("Programming tiles")
            res = pool.map(program_fpgas, params)

            if not all(res):
                logging.error("Could not program tiles!")
                self.properly_formed_station = False

        # Check if initialisation is required, and if so initialise
        if self.configuration['station']['initialise'] and self.properly_formed_station:
            logging.info("Initialising tiles")
            res = pool.map(initialise_tile, params)

            if not all(res):
                logging.error("Could not initialise tiles!")
                self.properly_formed_station = False

        # Ready from pool
        if pool is not None:
            pool.terminate()

        # Connect all tiles. If the tiles are not configured properly then the below calls will
        # fail, in which case set properly_formed_station to false
        try:
            for tile in self.tiles:
                tile.connect()
                tile.tpm_ada.initialise_adas()
        except Exception as e:
            self.properly_formed_station = False
            raise e

        # Initialise if required
        if self.configuration['station']['initialise'] and self.properly_formed_station:
            logging.info("Forming station")
            self._form_station()

            logging.info("Initializing Channelizer")
            center_channel = 282

            for i, tile in enumerate(self.tiles):
                tile['fpga1.dsp_regfile.channelizer_sel.lo_channel'] = center_channel
                tile['fpga2.dsp_regfile.channelizer_sel.lo_channel'] = center_channel

            for tile in self.tiles:
                tile.tpm.tpm_pattern_generator[0].initialise()
                tile.tpm.tpm_pattern_generator[1].initialise()

            # If in testing mode, override tile-specific test generators
            if self.configuration['station']['enable_test']:
                for tile in self.tiles:
                    for gen in tile.tpm.test_generator:
                        gen.channel_select(0x0000)
                        gen.disable_prdg()

                for tile in self.tiles:
                    for gen in tile.tpm.test_generator:
                        gen.set_tone(0, 139624023.438 + 1e3, 1)
                        gen.channel_select(0xFFFF)

            # Set ADA gain if required
            if self.configuration["station"]["ada_gain"] is not None:
                # Check if the provided file is valid

                if type(self.configuration["station"]["ada_gain"]) == int:
                    self.equalize_ada_gain(self.configuration["station"]["ada_gain"])
                elif not os.path.isfile(self.configuration["station"]["ada_gain"]):
                    logging.warning("Provided ada gain file is invalid ({})".format(self.configuration["station"]["ada_gain"]))
                else:
                    # Try loading file
                    with open(self.configuration["station"]["ada_gain"], 'r') as f:
                        try:
                            gains = yaml.load(f, yaml.FullLoader)
                            if not ('tpm_1' in gains and 'tpm_2' in gains):
                                logging.warning("Ada gains file should have entries for TPM 1 and TPM 2")

                            # Set gain
                            logging.info("Setting ADA gains")
                            for i, tile in enumerate(self.tiles):
                                tpm_gains = gains["tpm_{}".format(i + 1)]
                                tile.tpm_ada.initialise_adas()
                                if len(tpm_gains) == 1:
                                    tile.tpm_ada.set_ada_gain(tpm_gains[0])
                                elif len(tpm_gains) == 32:
                                    for j in range(32):
                                        tile.tpm_ada.set_ada_gain_spi(tpm_gains[j], j)
                                else:
                                    logging.warning("Invalid number of gains definde for TPM {}".format(i+1))

                        except Exception as e:
                            logging.warning("Could not apply ada gain: {}".format(e))

            # Otherwise perform automatic equalization
            else:
                self.equalize_ada_gain(16)

            # If initialising, synchronise all tiles in station
            logging.info("Synchronising station")
            self._station_post_synchronisation()
            self._synchronise_adc_clk()
            self._synchronise_ddc(sysref_period=1280)  # 1280 works for all supported frequencies
            self._synchronise_tiles(self.configuration['network']['lmc']['use_teng'])

        elif not self.properly_formed_station:
            logging.warning("Some tiles were not initialised or programmed. Not forming station")

        # If not initialising, check that station is formed properly
        else:
            self.check_station_status()

    def check_station_status(self):
        """ Check that the station is still valid """
        tile_ids = []
        for tile in self.tiles:
            if tile.tpm is None:
                self.properly_formed_station = False
                break

            tile_id = tile.get_tile_id()
            if tile.get_tile_id() < len(self.tiles) and tile_id not in tile_ids:
                tile_ids.append(tile_id)
            else:
                self.properly_formed_station = False
                break

        if not self.properly_formed_station:
            logging.warning("Station configuration is incorrect (unreachable TPMs or incorrect tile ids)!")

        return self.properly_formed_station

    def _form_station(self):
        """ Forms the station """
        # Assign station and tile id, and tweak transceivers
        for i, tile in enumerate(self.tiles):
            tile.set_station_id(self._station_id, i)
            tile.tweak_transceivers()

    def _synchronise_adc_clk(self):
        sampling_frequency = self.configuration['observation']['sampling_frequency']
        if sampling_frequency == 700e6:
            logging.info("Synchronising FPGA ADC Clock...")
            for tile in self.tiles:
                tile['fpga1.pps_manager.sync_tc_adc_clk'] = 0x7
                tile['fpga2.pps_manager.sync_tc_adc_clk'] = 0x7
            self.tiles[0].wait_pps_event2()
            for tile in self.tiles:
                for fpga in tile.tpm.tpm_fpga:
                    fpga.fpga_align_adc_clk(sampling_frequency)

    def _synchronise_ddc(self, sysref_period):
        """ Synchronise the NCO in the DDC on all ADCs of all tiles """
        logging.info("Synchronising DDC")
        for tile in self.tiles:
            tile.set_fpga_sysref_gen(sysref_period)

            tile['pll', 0x402] = 0x8 # 0xD0
            tile['pll', 0x403] = 0x0 # 0xA2
            tile['pll', 0x404] = 0x1 # 0x4
            tile['pll', 0xF] = 0x1
            while tile['pll', 0xF] & 0x1 == 0x1:
                time.sleep(0.1)

        ddc_frequency = self.configuration['observation']['ddc_frequency']
        sampling_frequency = self.configuration['observation']['sampling_frequency']
        for tile in self.tiles:
            for n in range(16):
                tile.tpm.tpm_adc[n].adc_single_start_dual_14_ddc(sampling_frequency=sampling_frequency,
                                                                 ddc_frequency=ddc_frequency,
                                                                 low_bitrate=True)

            for n in range(16):
                if n < 8:
                    tile['adc' + str(n), 0x120] = 0xA
                else:
                    tile['adc' + str(n), 0x120] = 0x1A

        self.tiles[0].wait_pps_event()

        for tile in self.tiles:
            for n in range(4):
                tile.tpm.tpm_jesd[n].jesd_core_restart()
        for tile in self.tiles:
            for n in range(4):
                tile.tpm.tpm_jesd[n].jesd_core_check()

        self.reset_ddc()

    def reset_ddc(self, tiles="all"):
        if tiles == "all":
            tiles = self.tiles
        for tile in tiles:
            tile.write_adc_broadcast(0x300, 0x1, 0)

        self.tiles[0].wait_pps_event()

        time.sleep(0.1)

        for tile in tiles:
            tile.write_adc_broadcast(0x300, 0x0, 1)

        while True:
            rd = self['board.spi.cmd']
            done = 1
            for s in rd:
                if s & 0x1 == 1:
                    done = 0
            if done == 1:
                break 

    def _synchronise_tiles(self, use_teng=False):
        """ Synchronise time on all tiles """

        pps_detect = self['fpga1.pps_manager.pps_detected']
        logging.debug("FPGA1 PPS detection register is ({})".format(pps_detect))
        pps_detect = self['fpga2.pps_manager.pps_detected']
        logging.debug("FPGA2 PPS detection register is ({})".format(pps_detect))

        # Repeat operation until Tiles are synchronised
        while True:
            # Read the current time on first tile
            self.tiles[0].wait_pps_event()

            time.sleep(0.2)

            # PPS edge detected, write time to all tiles
            curr_time = self.tiles[0].get_fpga_time(Device.FPGA_1)
            logging.info("Synchronising tiles in station with time %d" % curr_time)

            for tile in self.tiles:
                tile.set_fpga_time(Device.FPGA_1, curr_time)
                tile.set_fpga_time(Device.FPGA_2, curr_time)

            # All done, check that PPS on all boards are the same
            self.tiles[0].wait_pps_event()

            times = set()
            for tile in self.tiles:
                times.add(tile.get_fpga_time(Device.FPGA_1))
                times.add(tile.get_fpga_time(Device.FPGA_2))

            if len(times) == 1:
                break

        # Tiles synchronised
        curr_time = self.tiles[0].get_fpga_time(Device.FPGA_1)
        logging.info("Tiles in station synchronised, time is %d" % curr_time)

        # Set LMC data lanes
        for tile in self.tiles:
            # Configure standard data streams
            if use_teng:
                logging.info("Using 10G for LMC traffic")
                tile.set_lmc_download("10g", 8192,
                                      dst_ip=self.configuration['network']['lmc']['lmc_ip'],
                                      dst_port=self.configuration['network']['lmc']['lmc_port'],
                                      lmc_mac=self.configuration['network']['lmc']['lmc_mac'])
            else:
                # Configure integrated data streams
                logging.info("Using 1G for LMC traffic")
                tile.set_lmc_download("1g")
                
            # Configure integrated data streams
            if self.configuration['network']['lmc']['use_teng_integrated']:
                logging.info("Using 10G for integrated LMC traffic")
                tile.set_lmc_integrated_download("10g", 1024, 2048, 
                                                 dst_ip=self.configuration['network']['lmc']['lmc_ip'],
                                                 lmc_mac=self.configuration['network']['lmc']['lmc_mac'])
            else:
                # Configure integrated data streams
                logging.info("Using 1G for integrated LMC traffic")                
                tile.set_lmc_integrated_download("1g", 1024, 2048)
                

        # Start data acquisition on all boards
        delay = 2
        t0 = self.tiles[0].get_fpga_time(Device.FPGA_1)
        for tile in self.tiles:
            tile.start_acquisition(start_time=t0, delay=delay)

        t1 = self.tiles[0].get_fpga_time(Device.FPGA_1)
        if t0 + delay > t1:
            logging.info("Waiting for start acquisition")
            while self.tiles[0]['fpga1.dsp_regfile.stream_status.channelizer_vld'] == 0:
                time.sleep(0.1)
        else:
            logging.error("Start data acquisition not synchronised! Rerun initialisation")
            exit()

    def _station_post_synchronisation(self):
        """ Post tile configuration synchronization """
        for tile in self.tiles:
            tile['fpga1.pps_manager.sync_cnt_enable'] = 0x7
            tile['fpga2.pps_manager.sync_cnt_enable'] = 0x7
        time.sleep(0.2)
        for tile in self.tiles:
            tile['fpga1.pps_manager.sync_cnt_enable'] = 0x0
            tile['fpga2.pps_manager.sync_cnt_enable'] = 0x0

        # Station synchronisation loop
        sync_loop = 0
        max_sync_loop = 3
        while sync_loop < max_sync_loop:
            self.tiles[0].wait_pps_event2()

            current_tc = [tile.get_phase_terminal_count() for tile in self.tiles]
            delay = [tile.get_pps_delay() for tile in self.tiles]

            for n in range(len(self.tiles)):
                self.tiles[n].set_phase_terminal_count(self.tiles[n].calculate_delay(delay[n], current_tc[n],
                                                                                     16, 24))

            self.tiles[0].wait_pps_event2()

            current_tc = [tile.get_phase_terminal_count() for tile in self.tiles]
            delay = [tile.get_pps_delay() for tile in self.tiles]

            for n in range(len(self.tiles)):
                self.tiles[n].set_phase_terminal_count(self.tiles[n].calculate_delay(delay[n], current_tc[n],
                                                                                     delay[0] - 4, delay[0] + 4))

            self.tiles[0].wait_pps_event2()

            delay = [tile.get_pps_delay() for tile in self.tiles]

            synced = 1
            for n in range(len(self.tiles) - 1):
                if abs(delay[0] - delay[n + 1]) > 4:
                    logging.info("Resynchronizing station ({})".format(delay))
                    sync_loop += 1
                    synced = 0

            if synced == 1:
                phase1 = [hex(tile['fpga1.pps_manager.sync_phase']) for tile in self.tiles]
                phase2 = [hex(tile['fpga2.pps_manager.sync_phase']) for tile in self.tiles]
                logging.debug("Final FPGA1 clock phase ({})".format(phase1))
                logging.debug("Final FPGA2 clock phase ({})".format(phase2))

                logging.info("Finished station post synchronisation ({})".format(delay))
                return

        logging.error("Station post synchronisation failed!")

    def check_tiles_synchronization(self):
        # check FPGA time
        self.tiles[0].wait_pps_event()
        t0 = self.tiles[0]["fpga1.pps_manager.curr_time_read_val"]
        t1 = self.tiles[1]["fpga2.pps_manager.curr_time_read_val"]
        logging.info("FPGA1 time is " + str(t0))
        logging.info("FPGA2 time is " + str(t1))
        if t0 != t1:
            logging.warning("Time different between FPGAs detected!")

        # check FPGA timestamp
        t0 = self.tiles[0]["fpga1.pps_manager.timestamp_read_val"]
        t1 = self.tiles[1]["fpga1.pps_manager.timestamp_read_val"]
        logging.info("FPGA1 timestamp is " + str(t0))
        logging.info("FPGA2 timestamp is " + str(t1))
        if abs(t0 - t1) > 1:
            logging.warning("Timestamp different between FPGAs detected!")

    def equalize_ada_gain(self, required_rms=20):
        """ Equalize the ada gain to get target RMS"""

        # Set lowest gain
        self.set_ada_gain(-6)
        time.sleep(1)

        # Loop over all tiles
        for tt, tile in enumerate(self.tiles):

            # Get current RMS
            rms = [x for x in tile.get_adc_rms()]

            # Loop over all signals
            for channel in range(len(rms)):
                # Calculate required gain
                if rms[channel] / required_rms > 0:
                    gain = 20 * math.log10(required_rms / rms[channel])
                else:
                    gain = 0

                # Apply attenuation
                if gain < 0:
                    logging.warning("Not possible to set ADA gain lower than -6 dB for tile %d signal %d" % (tt, channel))
                elif gain > 21:
                    logging.warning("Not possible to set ADA gain higher than 15 dB for tile %d signal %d" % (tt, channel))
                    gain = 21 
                else:
                    tile.tpm.tpm_ada.set_ada_gain_spi(-6 + int(gain), channel)

    def set_ada_gain(self, attenuation):
        """ Set same preadu attenuation in all preadus """
        # Loop over all tiles
        for tile in self.tiles:
            tile.tpm.tpm_ada.set_ada_gain(attenuation)

    # ------------------------------------------------------------------------------------------------
    def calculate_center_frequency(self):
        decimation_ratio = 8
        sampling_frequency = self.configuration['observation']['sampling_frequency'] / decimation_ratio
        ddc_frequency = self.configuration['observation']['ddc_frequency']
        return int(ddc_frequency / sampling_frequency * 1024) / 1024.0 * sampling_frequency

    def test_generator_set_tone(self, dds, frequency=100e6, ampl=0.0, phase=0.0, delay=512):
        decimation_ratio = 8
        sampling_frequency = self.configuration['observation']['sampling_frequency']
        ddc_frequency = self.configuration['observation']['ddc_frequency']
        translated_frequency = frequency - ddc_frequency + sampling_frequency / (decimation_ratio * 4.0)

        t0 = self.tiles[0]["fpga1.pps_manager.timestamp_read_val"] + delay
        for tile in self.tiles:
            for gen in tile.tpm.test_generator:
                gen.set_tone(dds, translated_frequency, ampl, phase, t0)

        t1 = self.tiles[0]["fpga1.pps_manager.timestamp_read_val"]
        if t1 > t0:
            logging.info("Set tone test pattern generators synchronisation failed.")

    def test_generator_disable_tone(self, dds, delay=512):
        t0 = self.tiles[0]["fpga1.pps_manager.timestamp_read_val"] + delay
        for tile in self.tiles:
            for gen in tile.tpm.test_generator:
                gen.set_tone(dds, 0, 0, 0, t0)
        t1 = self.tiles[0]["fpga1.pps_manager.timestamp_read_val"]
        if t1 > t0:
            logging.info("Set tone test pattern generators synchronisation failed.")

    def test_generator_set_noise(self, ampl=0.0, delay=512):
        t0 = self.tiles[0]["fpga1.pps_manager.timestamp_read_val"] + delay
        for tile in self.tiles:
            for gen in tile.tpm.test_generator:
                gen.enable_prdg(ampl, t0)
        t1 = self.tiles[0]["fpga1.pps_manager.timestamp_read_val"]
        if t1 > t0:
            logging.info("Set tone test pattern generators synchronisation failed.")

    def test_generator_input_select(self, inputs):
        for tile in self.tiles:
            tile.test_generator[0].channel_select(inputs & 0xFFFF)
            tile.test_generator[1].channel_select((inputs >> 16) & 0xFFFF)

    def enable_channeliser_test_pattern(self):
        """ Enable channeliser test pattern """
        for tile in self.tiles:
            tile.tpm.tpm_pattern_generator[0].start_pattern("channel")
            tile.tpm.tpm_pattern_generator[1].start_pattern("channel")

    def stop_channeliser_test_pattern(self):
        """ Enable channeliser test pattern """
        for tile in self.tiles:
            tile.tpm.tpm_pattern_generator[0].stop_pattern("channel")
            tile.tpm.tpm_pattern_generator[1].stop_pattern("channel")

    def spead_tx_disable(self):
        for tile in self.tiles:
            tile['fpga1.dsp_regfile.spead_tx_enable']=0
            tile['fpga2.dsp_regfile.spead_tx_enable']=0

    # ------------------------------------------------------------------------------------------------

    def mii_test(self, pkt_num):
        """ Perform mii test """

        for i, tile in enumerate(self.tiles):
            logging.debug("MII test setting Tile " + str(i))
            tile.mii_prepare_test(i + 1)

        for i, tile in enumerate(self.tiles):
            logging.debug("MII test starting Tile " + str(i))
            tile.mii_exec_test(pkt_num, wait_result=False)

        while True:
            for i, tile in enumerate(self.tiles):
                logging.debug("Tile " + str(i) + " MII test result:")
                tile.mii_show_result()
                k = raw_input("Enter quit to exit. Any other key to continue.")
                if k == "quit":
                    return

    def enable_adc_trigger(self, threshold=127):
        """ Enable ADC trigger to send raw data when an RMS threshold is reached"""

        if 0 > threshold > 127:
            logging.error("Invalid threshold, must be 1 - 127")
            return

        # Enable trigger
        self['fpga1.lmc_gen.raw_ext_trigger_enable'] = 1
        self['fpga2.lmc_gen.raw_ext_trigger_enable'] = 1

        # Set threshold
        for tile in self.tiles:
            for adc in tile.tpm.tpm_adc:
                adc.adc_set_fast_detect(threshold << 6)

    def disable_adc_trigger(self):
        """ Disable ADC trigger """
        for tile in self.tiles:
            tile['fpga1.lmc_gen.raw_ext_trigger_enable'] = 0
            tile['fpga2.lmc_gen.raw_ext_trigger_enable'] = 0

    def set_channelizer_truncation(self, trunc):
        for tile in self.tiles:
            tile.set_channeliser_truncation(trunc)

    # ------------------------------------ DATA OPERATIONS -------------------------------------------

    def send_raw_data(self, sync=False, period=0, timeout=0):
        """ Send raw data from all Tiles """
        self._wait_available()
        t0 = self.tiles[0].get_fpga_timestamp(Device.FPGA_1)
        for tile in self.tiles:
            tile.send_raw_data(sync=sync, period=period, timeout=timeout, timestamp=t0, seconds=self._seconds)
        return self._check_data_sync(t0)

    def send_raw_data_synchronised(self, period=0, timeout=0):
        """ Send synchronised raw data from all Tiles """
        self._wait_available()
        t0 = self.tiles[0].get_fpga_timestamp(Device.FPGA_1)
        for tile in self.tiles:
            tile.send_raw_data_synchronised(period=period, timeout=timeout, timestamp=t0, seconds=self._seconds)
        return self._check_data_sync(t0)

    def stop_data_transmission(self):
        """ Stop data transmission """
        for tile in self.tiles:
            tile.stop_data_transmission()

    def stop_integrated_data(self):
        """ Stop integrated data transmission """
        for tile in self.tiles:
            tile.stop_integrated_data()

    def _wait_available(self):
        """ Make sure all boards can send data """
        while any([tile.check_pending_data_requests() for tile in self.tiles]):
            logging.info("Waiting for pending data requests to finish")
            time.sleep(0.1)

    def _check_data_sync(self, t0):
        """ Check whether data synchronisation worked """
        delay = self._seconds * (1 / (40960 * 1e-9) / 256)
        timestamps = [tile.get_fpga_timestamp(Device.FPGA_1) for tile in self.tiles]
        logging.debug("Data sync check: timestamp={}, delay={}".format(str(timestamps), delay))
        return all([(t0 + delay) > t1 for t1 in timestamps])

    def test_wr_exec(self):
        import time
        start = time.time()
        ba = self.tiles[0].tpm.register_list['%s.pattern_gen.%s_data' % ('fpga1', "beamf")]['address']

        for n in range(1024):
            self.tiles[0][ba] = range(256)

        end = time.time()
        logging.debug("test_wr_exec: {}".format((end - start)))

    # ------------------------------------------- TEST FUNCTIONS ---------------------------------------

    def test_rd_exec(self):
        import time
        start = time.time()
        ba = self.tiles[0].tpm.register_list['%s.pattern_gen.%s_data' % ('fpga1', "beamf")]['address']

        for n in range(1024):
            self.tiles[0].tpm.read_register(ba, n=256)

        end = time.time()
        logging.debug("test_rd_exec: {}".format((end - start)))

    def ddr3_check(self):
        try:
            while True:
                for n in range(len(self.tiles)):
                    if (self.tiles[n]['fpga1.ddr3_if.status'] & 0x100) != 256:
                        logging.info("Tile" + str(n) + " FPGA1 DDR Error Detected!")
                        logging.info(hex(self.tiles[n]['fpga1.ddr3_if.status']))
                        logging.info(time.asctime(time.localtime(time.time())))
                        time.sleep(5)

                    if (self.tiles[n]['fpga2.ddr3_if.status'] & 0x100) != 256:
                        logging.info("Tile" + str(n) + " FPGA2 DDR Error Detected!")
                        logging.info(hex(self.tiles[n]['fpga2.ddr3_if.status']))
                        logging.info("localtime={}".format(time.asctime(time.localtime(time.time()))))
                        time.sleep(5)
        except KeyboardInterrupt:
            pass

    def check_adc_sysref(self):
        for adc in range(16):
            error = 0
            values = self['adc' + str(adc), 0x128]
            for i in range(len(values)):
                msb = (values[i] & 0xF0) >> 4
                lsb = (values[i] & 0x0F) >> 0
                if msb == 0 and lsb <= 7:
                    logging.warning('Possible setup error in tile %d adc %d' % (i, adc))
                    error = 1
                if msb >= 9 and lsb == 0:
                    logging.warning('Possible hold error in tile %d adc %d' % (i, adc))
                    error = 1
                if msb == 0 and lsb == 0:
                    logging.warning('Possible setup and hold error in tile %d adc %d' % (i, adc))
                    error = 1
            if error == 0:
                logging.info('ADC %d sysref OK!' % adc)

    def temperature_check(self):
        t = 0
        pll_reset = 0
        while True:
            try:
                t = station.tiles[0].get_temperature()
            except:
                pass
            if t > 60:
                station['pll',0x0] = 0x81
                pll_reset = 1
            if pll_reset == 1:
                print("reset PLL")
            time.sleep(1)

    # ------------------------------------------- OVERLOADED FUNCTIONS ---------------------------------------

    def __getitem__(self, key):
        """ Read register across all tiles """
        return [tile.tpm[key] for tile in self.tiles]

    def __setitem__(self, key, value):
        """ Write register across all tiles """
        for tile in self.tiles:
            tile.tpm[key] = value


def apply_config_file(input_dict, output_dict):
    """ Recursively copy value from input_dict to output_dict"""
    for k, v in input_dict.items():
        if type(v) is dict:
            apply_config_file(v, output_dict[k])
        elif k not in output_dict.keys():
            logging.warning("{} not a valid configuration item. Skipping".format(k))
        else:
            output_dict[k] = v


def load_configuration_file(filepath):
    """ Load station configuration from configuration file """
    if filepath is not None:
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            logging.error("Specified configuration file ({}) does not exist. Exiting".format(filepath))
            exit()

        # Configuration file defined, load and update default configuration
        with open(filepath, 'r') as f:
            c = yaml.load(f, yaml.FullLoader)
            apply_config_file(c, configuration)

            # Fix beam bandwidth and start frequency (in case they were written in scientific notation)
            configuration['observation']['bandwidth'] = \
                float(configuration['observation']['bandwidth'])
            configuration['observation']['ddc_frequency'] = \
                float(configuration['observation']['ddc_frequency'])

    else:
        logging.error("No configuration file specified. Exiting")
        exit()


def load_station_configuration(config_params):
    """ Combine configuration defined in configuration file with command-line arguments """

    # If a configuration file is defined, check if it exists and load it
    load_configuration_file(config_params.config)

    # Go through command line options and update where necessary
    if config_params.bandwidth is not None:
        configuration['observation']['bandwidth'] = config_params.bandwidth
    if config_params.bitfile is not None:
        configuration['station']['bitfile'] = config_params.bitfile
    if config_params.chan_trunc is not None:
        configuration['station']['channel_truncation'] = config_params.chan_trunc
    if config_params.channel_integ is not None:
        configuration['station']['channel_integration_time'] = config_params.channel_integ
    if config_params.enable_test is not None:
        configuration['station']['enable_test'] = config_params.enable_test
    if config_params.initialise is not None:
        configuration['station']['initialise'] = config_params.initialise
    if config_params.lmc_ip is not None:
        configuration['network']['lmc']['lmc_ip'] = config_params.lmc_ip
    if config_params.lmc_mac is not None:
        configuration['network']['lmc']['lmc_mac'] = config_params.lmc_mac
    if config_params.lmc_port is not None:
        configuration['network']['lmc']['lmc_port'] = config_params.lmc_port
    if config_params.port is not None:
        configuration['network']['lmc']['tpm_cpld_port'] = config_params.port
    if config_params.program is not None:
        configuration['station']['program'] = config_params.program
    if config_params.program_cpld is not None:
        configuration['station']['program_cpld'] = config_params.program_cpld
    if config_params.ddc_frequency is not None:
        configuration['observation']['ddc_frequency'] = config_params.ddc_frequency
    if config_params.sampling_frequency is not None:
        configuration['observation']['sampling_frequency'] = config_params.sampling_frequency
    if config_params.tiles is not None:
        configuration['tiles'] = config_params.tiles.split(',')
    if config_params.use_teng is not None:
        configuration['network']['lmc']['use_teng'] = config_params.use_teng

    return configuration


if __name__ == "__main__":
    from optparse import OptionParser
    from sys import argv, stdout

    parser = OptionParser(usage="usage: %station [options]")
    parser.add_option("--config", action="store", dest="config",
                      type="str", default=None, help="Configuration file [default: None]")
    parser.add_option("--port", action="store", dest="port",
                      type="int", default=None, help="Port [default: None]")
    parser.add_option("--lmc_ip", action="store", dest="lmc_ip",
                      default=None, help="IP [default: None]")
    parser.add_option("--lmc_port", action="store", dest="lmc_port",
                      type="int", default=None, help="Port [default: None]")
    parser.add_option("--lmc-mac", action="store", dest="lmc_mac",
                      type="int", default=None, help="LMC MAC address [default: None]")
    parser.add_option("-f", "--bitfile", action="store", dest="bitfile",
                      default=None, help="Bitfile to use (-P still required) [default: None]")
    parser.add_option("-t", "--tiles", action="store", dest="tiles",
                      default=None, help="Tiles to add to station [default: None]")
    parser.add_option("-P", "--program", action="store_true", dest="program",
                      default=False, help="Program FPGAs [default: False]")
    parser.add_option("-I", "--initialise", action="store_true", dest="initialise",
                      default=False, help="Initialise TPM [default: False]")
    parser.add_option("-C", "--program_cpld", action="store_true", dest="program_cpld",
                      default=False, help="Update CPLD firmware (requires -f option)")
    parser.add_option("-E", "--equalize-signals", action="store_true", dest="equalize_signals",
                      default=False, help="Equalize ADC signals")
    parser.add_option("-T", "--enable-test", action="store_true", dest="enable_test",
                      default=False, help="Enable test pattern (default: False)")
    parser.add_option("--use_teng", action="store_true", dest="use_teng",
                      default=None, help="Use 10G for LMC (default: None)")
    parser.add_option("--chan-trunc", action="store", dest="chan_trunc",
                      default=None, type="int", help="Channeliser truncation [default: None]")
    parser.add_option("--channel-integration-time", action="store", dest="channel_integ",
                      type="float", default=None, help="Integrated channel integration time [default: None]")
    parser.add_option("--bandwidth", action="store", dest="bandwidth",
                      type="float", default=None, help="Channelizer bandwidth [default: None]")
    parser.add_option("--ddc_frequency", action="store", dest="ddc_frequency",
                      type="float", default=None, help="DDC frequency [default: None]")
    parser.add_option("--sampling_frequency", action="store", dest="sampling_frequency",
                      type="float", default=700e6, help="ADC sampling frequency. Supported frequency are 700e6, 800e6 [default: 700e6]")
    (conf, args) = parser.parse_args(argv[1:])

    # Set logging
    log = logging.getLogger('')
    log.setLevel(logging.INFO)
    line_format = logging.Formatter("%(asctime)s - %(levelname)s - %(threadName)s - %(message)s")
    ch = logging.StreamHandler(stdout)
    ch.setFormatter(line_format)
    log.addHandler(ch)

    # Set current thread name
    threading.current_thread().name = "Station"

    # Load station configuration
    configuration = load_station_configuration(conf)

    # Create station
    station = Station(configuration)

    # Connect station (program, initialise and configure if required)
    station.connect()

    # Equalize signals if required
    if conf.equalize_signals:
        station.equalize_ada_gain(16)

    for tile in station.tiles:
        tile.set_channeliser_truncation(configuration['station']['channel_truncation'])

#    for tile in station.tiles:
#        for gen in tile.tpm.test_generator:
#            gen.channel_select(0x0000)
#            gen.disable_prdg()
#
#    for tile in station.tiles:
#        for gen in tile.tpm.test_generator:
#            gen.set_tone(0, 139624023.438 + 1e3, 2)
#            gen.channel_select(0xFFFF)

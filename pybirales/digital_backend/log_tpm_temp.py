#! /usr/bin/env python
from sys import exit
import numpy as np
import logging
import time
import datetime

from pybirales.digital_backend.tile_debris import Tile
from pyfabil import Device
from pybirales.digital_backend.digital_backend import Station, load_station_configuration

from multiprocessing import Pool
from threading import Thread
import threading
import logging
import yaml
import time
import math
import sys
import os

import calendar


def dt_to_timestamp(d):
    return calendar.timegm(d.timetuple())


def ts_to_datestring(tstamp, formato="%Y-%m-%d %H:%M:%S"):
    return datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(tstamp), formato)


# Define default configuration
configuration = {'tiles': None,
                 'station': {
                     'id': 0,
                     'name': "debris",
                     "number_of_antennas": 32,
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
                         'use_teng_integrated': True},
                     'csp_ingest': {
                         'src_ip': "10.0.10.254",
                         'dst_mac': 0x248A078F9D38,
                         'src_port': None,
                         'dst_port': 4660,
                         'dst_ip': "10.0.10.200",
                         'src_mac': None}
                 }
                 }

__author__ = 'Andrea Mattana'

# Global tile reference
temp_dir = "/opt/tpm_temperatures/"

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
                      type="float", default=700e6,
                      help="ADC sampling frequency. Supported frequency are 700e6, 800e6 [default: 700e6]")
    (conf, args) = parser.parse_args(argv[1:])
    # Load station configuration
    configuration = load_station_configuration(conf)
    # Create station
    station = Station(configuration)
    # Connect station (program, initialise and configure if required)
    station.connect()
    tile = station.tiles[0]
    fname = datetime.datetime.strftime(datetime.datetime.utcnow(), "%Y-%m-%d_%H%M%S.txt")
    outfile = temp_dir + fname
    print("Writing file: " + outfile)
    try:
        with open(outfile, "w") as f:
            print("Date\tTime\tBoard\tFPGA0\tFPGA1")
            f.write("Date\tTime\tBoard\tFPGA0\tFPGA1\n")
            f.flush()
            while True:
                tstamp = dt_to_timestamp(datetime.datetime.utcnow())
                orario = ts_to_datestring(tstamp, "%Y-%m-%d\t%H:%M:%S")
                msg = "%d\t%s\t%3.1f\t%3.1f\t%3.1f\n" % (tstamp, orario, tile.get_temperature(),
                                                         tile.get_fpga0_temperature(), tile.get_fpga1_temperature())
                f.write(msg)
                f.flush()
                print(msg[:-1])
                time.sleep(5)
    except:
        print("Disconnected from TPM")

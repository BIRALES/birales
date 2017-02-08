#!/usr/bin/python

import logging
import os

from pybirales.modules.aavs_channel_reader import AAVSChannelReader
from pybirales.modules.correlator import Correlator
from pybirales.base import settings
from pybirales.base.pipeline_manager import PipelineManager
from pybirales.modules.channeliser import PFB
from pybirales.modules.terminator import Terminator


# ------------------------------------------- Pipelines ---------------------------------------------

def aavs_correlator(manager):
    reader = AAVSChannelReader(settings.aavs_channel_reader)
    pfb = PFB(settings.channeliser, reader.output_blob)
    correlator = Correlator(settings.correlator, pfb.output_blob)
    terminator = Terminator(settings.terminator, correlator.output_blob)

    manager.add_module("reader", reader)
    manager.add_module("pfb", pfb)
    manager.add_module("correlator", correlator)
    manager.add_module("terminator", terminator)

if __name__ == "__main__":

    # Use OptionParse to get command-line arguments
    from optparse import OptionParser
    from sys import argv

    # Define parameters
    parser = OptionParser(usage="usage: aavs_correlator.py CONFIG_FILE [options]")

    # Check number of command-line arguments
    if len(argv) < 2:
        print("Configuration file required. Uage: aavs_correlator.py CONFIG_FILE [options]")
        exit()

    # Check if configuration file was passed
    if not os.path.exists(argv[1]):
        print("Configuration file required. Uage: aavs_correlator.py CONFIG_FILE [options]")
        exit()

    # Parse command-line arguments
    (conf, args) = parser.parse_args(argv[2:])

    # Create pipeline manager
    manager = PipelineManager(argv[1])

    logging.info("Initialising")

    aavs_correlator(manager)

    logging.info("Started")

    # Start pipeline
    manager.start_pipeline()

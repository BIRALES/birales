#!/usr/bin/python

import logging
import os

from pybirales.base import settings
from pybirales.base.pipeline_manager import PipelineManager
from pybirales.modules.continuous_channel_receiver import ContinuousChannelReceiver
from pybirales.modules.beamformer import Beamformer
from pybirales.modules.terminator import Terminator


# ------------------------------------------- Pipelines ---------------------------------------------

def pharos_pipeline(manager):
    receiver = ContinuousChannelReceiver(settings.continuous_channel_receiver)
    beamformer = Beamformer(settings.beamformer, receiver.output_blob)
    terminator = Terminator(settings.terminator, beamformer.output_blob)

    manager.add_module("reader", receiver)
    manager.add_module("beamformer", beamformer)
    manager.add_module("terminator", terminator)


if __name__ == "__main__":

    # Use OptionParse to get command-line arguments
    from optparse import OptionParser
    from sys import argv

    # Define parameters
    parser = OptionParser(usage="usage: pharos.py CONFIG_FILE [options]")

    # Check number of command-line arguments
    if len(argv) < 2:
        print("Configuration file required. Usage: pharos.py CONFIG_FILE [options]")
        exit()

    # Check if configuration file was passed
    if not os.path.exists(argv[1]):
        print("Configuration file required. Usage: pharos.py CONFIG_FILE [options]")
        exit()

    # Parse command-line arguments
    (conf, args) = parser.parse_args(argv[2:])

    # Create pipeline manager
    manager = PipelineManager(argv[1])

    logging.info("Initialising")

    pharos_pipeline(manager)

    logging.info("Started")

    # Start pipeline
    manager.start_pipeline()

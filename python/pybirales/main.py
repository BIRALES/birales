#!/usr/bin/python

import logging
import os

from modules.correlator import Correlator
from pybirales.modules.terminator import Terminator
from pybirales.plotters.beam_plotter import BeamformedDataPlotter
from pybirales.base.pipeline_manager import PipelineManager
from pybirales.base import settings
from pybirales.modules.beamformer import Beamformer
from pybirales.modules.channeliser import PFB
from pybirales.modules.generator import DummyDataGenerator
from pybirales.modules.persister import Persister
from pybirales.modules.receiver import Receiver
from pybirales.plotters.bandpass_plotter import BandpassPlotter
from pybirales.plotters.antenna_plotter import AntennaPlotter
from pybirales.plotters.channel_plotter import ChannelisedDataPlotter
from pybirales.plotters.raw_data_plotter import RawDataPlotter
from pybirales.plotters.raw_data_grid_plotter import RawDataGridPlotter

# ------------------------------------------- Pipelines ---------------------------------------------

def aavs_correlator(manager):
    generator = DummyDataGenerator(settings.generator)
    correlator = Correlator(settings.correlator, generator.output_blob)
    terminator = Terminator(settings.terminator, correlator.output_blob)

    manager.add_module("generator", generator)
    manager.add_module("correlator", correlator)
    manager.add_module("terminator", terminator)

def standalone_test(manager):
    generator = DummyDataGenerator(settings.generator)
    beamformer = Beamformer(settings.beamformer, generator.output_blob)
    pfb = PFB(settings.channeliser, beamformer.output_blob)
    terminator = Terminator(settings.terminator, pfb.output_blob)

    manager.add_module("generator", generator)
    manager.add_module("beamformer", beamformer)
    manager.add_module("pfb", pfb)
    manager.add_module("terminator", terminator)

    manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, pfb.output_blob)


def test_receiver(manager):
    receiver = Receiver(settings.receiver)
    ppf = PFB(settings.channeliser, receiver.output_blob)
    terminator = Terminator(None, ppf.output_blob)

    manager.add_module("receiver", receiver)
    manager.add_module("ppf", ppf)
    manager.add_module("terminator", terminator)

    #manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, ppf.output_blob)
    #manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)
    manager.add_plotter("antenna_plotter", AntennaPlotter, settings.antennaplotter, receiver.output_blob)


def birales_pipeline(manager):
    # Generate processing modules and data blobs
    receiver = Receiver(settings.receiver)
    beamformer = Beamformer(settings.beamformer, receiver.output_blob)
    ppf = PFB(settings.channeliser, beamformer.output_blob)
    persister = Persister(settings.persister, ppf.output_blob)

    # Add modules to pipeline manager
    manager.add_module("receiver", receiver)
    manager.add_module("beamformer", beamformer)
    manager.add_module("ppf", ppf)
    manager.add_module("persister", persister)

    # Add plotters
    manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)
    #manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, ppf.output_blob)

if __name__ == "__main__":

    # Use OptionParse to get command-line arguments
    from optparse import OptionParser
    from sys import argv, stdout

    # Define parameters
    parser = OptionParser(usage="usage: birales.py CONFIG_FILE [options]")

    # Check number of command-line arguments
    if len(argv) < 2:
        print("Configuration file required. Uage: birales.py CONFIG_FILE [options]")
        exit()

    # Check if configuration file was passed
    if not os.path.exists(argv[1]):
        print("Configuration file required. Uage: birales.py CONFIG_FILE [options]")
        exit()

    # Parse command-line arguments
    (conf, args) = parser.parse_args(argv[2:])

    # Create pipeline manager
    manager = PipelineManager(argv[1])

    logging.info("Initialising")

    #birales_pipeline(manager)
    #test_receiver(manager)
    #standalone_test(manager)
    aavs_correlator(manager)

    logging.info("Started")

    # Start pipeline
    manager.start_pipeline()
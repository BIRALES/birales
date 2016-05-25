#!/usr/bin/python

import logging

from modules.terminator import Terminator
from plotters.beam_plotter import BeamformedDataPlotter
from pybirales.base import settings
from pybirales.modules.beamformer import Beamformer
from pybirales.modules.channeliser import PFB
from pybirales.modules.generator import DummyDataGenerator
from pybirales.modules.persister import Persister
from pybirales.modules.receiver import Receiver
from pybirales.pipeline_manager import PipelineManager
from pybirales.plotters.bandpass_plotter import BandpassPlotter
from pybirales.plotters.channel_plotter import ChannelisedDataPlotter
from pybirales.plotters.raw_data_plotter import RawDataPlotter
from pybirales.plotters.raw_data_grid_plotter import RawDataGridPlotter


def test_receiver(manager):
    receiver = Receiver(settings.receiver)
    ppf = PFB(settings.channeliser, receiver.output_blob)
    terminator = Terminator(None, ppf.output_blob)

    manager.add_module("receiver", receiver)
    manager.add_module("ppf", ppf)
    manager.add_module("terminator", terminator)

    #manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, ppf.output_blob)
    manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)


def birales_pipeline(manager):
    # Generate processing modules and data blobs
    receiver = Receiver(settings.receiver)
    beamformer = Beamformer(settings.beamformer, receiver.output_blob)
    ppf = PFB(settings.channeliser, beamformer.output_blob)
    terminator = Terminator(None, ppf.output_blob)
    #persister = Persister(settings.persister, ppf.output_blob)

    # Add modules to pipeline manager
    manager.add_module("receiver", receiver)
    manager.add_module("beamformer", beamformer)
    manager.add_module("ppf", ppf)
    manager.add_module("terminator", terminator)
    #manager.add_module("persister", persister)

    # Add plotters
    manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)
#    manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, ppf.output_blob)

if __name__ == "__main__":

    # Create pipeline manager
    manager = PipelineManager("birales.ini")

    logging.info("Initialising")

    #birales_pipeline(manager)
    test_receiver(manager)

    logging.info("Started")

    # Start pipeline
    manager.start_pipeline()

# To install
# astropy
# astroplan

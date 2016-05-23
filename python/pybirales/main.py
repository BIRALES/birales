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


def test_receiver(manager):
    receiver = Receiver(settings.receiver)
    manager.add_module("receiver", receiver)


def test_channel_plotter(manager):
    # Generate processing modules and data blobs
   # receiver = Receiver(settings.receiver)
    generator = DummyDataGenerator(settings.generator)
    beamformer = Beamformer(settings.beamformer, generator.output_blob)
 #   ppf = PFB(settings.channeliser, beamformer.output_blob)
    terminator = Terminator(None, beamformer.output_blob)
    # persister = Persister(settings.persister, ppf.output_blob)

    # Add modules to pipeline manager
   # manager.add_module("receiver", receiver)
    manager.add_module("generator", generator)
    manager.add_module("beamformer", beamformer)
 #   manager.add_module("ppf", ppf)
    manager.add_module("terminator", terminator)
   # manager.add_module("persister", persister)

    # Add plotters
  #  manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)
  #  manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, ppf.output_blob)
  #  manager.add_plotter("raw_plotter", RawDataPlotter, settings.rawplotter, generator.output_blob)
  #  manager.add_plotter("beam_plotter", BeamformedDataPlotter, settings.beamplotter, beamformer.output_blob)


if __name__ == "__main__":

    # Create pipeline manager
    manager = PipelineManager("birales.ini")

    logging.info("Initialising")

    test_channel_plotter(manager)

    logging.info("Started")

    # Start pipeline
    manager.start_pipeline()

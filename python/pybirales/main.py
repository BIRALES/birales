#!/usr/bin/python

import logging

from modules.terminator import Terminator
from pybirales.base import settings
from pybirales.modules.beamformer import Beamformer
from pybirales.modules.channeliser import PFB
from pybirales.modules.generator import DummyDataGenerator
from pybirales.modules.persister import Persister
from pybirales.modules.receiver import Receiver
from pybirales.pipeline_manager import PipelineManager
from pybirales.plotters.raw_data_plotter import RawDataPlotter


def test_receiver(manager):
    receiver = Receiver(settings.receiver)
    manager.add_module("receiver", receiver)


def test_plotter(manager):
    generator = DummyDataGenerator(settings.generator)
    manager.add_module("generator", generator)
    terminator = Terminator(None, generator.output_blob)
    manager.add_module("terminator", terminator)

    raw_plotter = RawDataPlotter(settings.rawplotter, generator.output_blob)
    manager.add_plotter("raw_plotter", raw_plotter)


def test_pipeline(manager):
    # Generate processing modules and data blobs
    dummy_generator = DummyDataGenerator(settings.generator)
    beamformer = Beamformer(settings.beamformer, dummy_generator.output_blob)
    ppf = PFB(settings.channeliser, beamformer.output_blob)
    terminator = Terminator(None, ppf.output_blob)

    # Add modules to pipeline manager
    manager.add_module("dummy_generator", dummy_generator)
    manager.add_module("beamformer", beamformer)
    manager.add_module("ppf", ppf)
    manager.add_module("terminator", terminator)

if __name__ == "__main__":

    # Create pipeline manager
    manager = PipelineManager("birales.ini")

    logging.info("Initialising")

    test_plotter(manager)

    logging.info("Started")

    # Start pipeline
    manager.start_pipeline()
    manager.wait_pipeline()

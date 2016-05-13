#!/usr/bin/python

import logging
import time
from sys import stdout

from modules.terminator import Terminator
from pybirales.base import settings
from pybirales.modules.beamformer import Beamformer
from pybirales.modules.channeliser import PFB
from pybirales.modules.dummy import DummyDataGenerator
from pybirales.modules.persister import Persister
from pybirales.modules.receiver import Receiver
from pybirales.pipeline_manager import PipelineManager


def test_receiver(manager):
    # Generate processing modules and data blobs
    receiver = Receiver(settings.receiver)
    terminator = Terminator(None, receiver.output_blob)

    # Add modules to pipeline manager
    manager.add_module("receiver", receiver)
    manager.add_module("terminator", terminator)


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


def test_with_persister(manager):
    # Generate processing modules and data blobs
    dummy_generator = DummyDataGenerator(settings.generator)
    beamformer = Beamformer(settings.beamformer, dummy_generator.output_blob)
    ppf = PFB(settings.channeliser, beamformer.output_blob)
    persister = Persister(settings.persister, ppf.output_blob)

    # Add modules to pipeline manager
    manager.add_module("dummy_generator", dummy_generator)
    manager.add_module("beamformer", beamformer)
    manager.add_module("ppf", ppf)
    manager.add_module("persister", persister)

if __name__ == "__main__":

    # Create pipeline manager
    manager = PipelineManager("birales.ini")

    logging.info("Initialising")

    test_receiver(manager)

    logging.info("Started")

    # Start pipeline
    manager.start_pipeline()
    manager.wait_pipeline()

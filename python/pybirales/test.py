import logging
import time
from sys import stdout

from pybirales.base import settings
from pybirales.modules.beamformer import Beamformer
from pybirales.modules.channeliser import PFB
from pybirales.modules.dummy import DummyDataGenerator
from pybirales.pipeline_manager import PipelineManager

if __name__ == "__main__":

    # Create pipeline manager
    manager = PipelineManager("birales.ini")

    logging.info("Started")

    # Generate processing modules and data blobs
    dummy_generator = DummyDataGenerator(settings.generator)
    beamformer = Beamformer(settings.beamformer, dummy_generator.output_blob)
    ppf = PFB(settings.channeliser, beamformer.output_blob)

    # Add modules to pipeline manager
    manager.add_module("dummy_generator", dummy_generator)
    manager.add_module("beamformer", beamformer)
    manager.add_module("ppf", ppf)

    # Start pipeline
    manager.start_pipeline()
    manager.wait_pipeline()

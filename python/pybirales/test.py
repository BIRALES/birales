import time

from pybirales.modules.beamformer import Beamformer
from pybirales.modules.channeliser import PPF
from pybirales.modules.dummy import DummyDataGenerator
from pybirales.pipeline_manager import PipelineManager

if __name__ == "__main__":

    # Create pipeline manager
    manager = PipelineManager()

    # Processing modules configuration
    generator_config = {'nants': 32, 'nsamp': 131072, 'nchans': 1}
    ppf_config = {'nchans': 512}
    beamformer_config = {'nbeams': 32}

    # Generate processing modules and data blobs
    dummy_generator = DummyDataGenerator(generator_config)
    beamformer = Beamformer(beamformer_config, dummy_generator.output_blob)
    ppf = PPF(ppf_config, beamformer.output_blob)

    # Add modules to pipeline manager
    manager.add_module("dummy_generator", dummy_generator)
    manager.add_module("beamformer", beamformer)
    manager.add_module("ppf", ppf)

    # Start pipeline
    manager.start_pipeline()
    time.sleep(100)
    manager.stop_pipeline()

from pybirales.birales_config import BiralesConfig

from pybirales import settings
from pybirales.pipeline.modules.channeliser import PFB
from pybirales.pipeline.modules.correlator import Correlator
from pybirales.pipeline.modules.generators.generator import DummyDataGenerator
from pybirales.pipeline.modules.terminator import Terminator

from pybirales.pipeline.pipeline import PipelineManagerBuilder
from pybirales.tests.test_persister import TestPersister


class DummyGpuPipeline(PipelineManagerBuilder):
    def __init__(self):
        PipelineManagerBuilder.__init__(self)

        self.manager.name = 'Dummy GPU Correlator Pipeline'

        self._id = 'dummy_corr_pipeline_builder'

    def build(self):
        """
        This script runs the test receiver pipeline,
        using the specified CONFIGURATION.
        """

        # Initialise the modules
        receiver = DummyDataGenerator(settings.generator)
        pfb = PFB(settings.channeliser, receiver.output_blob)
        correlator = Correlator(settings.correlator, pfb.output_blob)
        persister = TestPersister(None, correlator.output_blob)
        terminator = Terminator(None, persister.output_blob)

        # Add modules to pipeline manager
        self.manager.add_module("receiver", receiver)
        self.manager.add_module("channeliser", pfb)
        self.manager.add_module("correlator", correlator)
        self.manager.add_module("persister", persister)
        self.manager.add_module("terminator", terminator)


if __name__ == "__main__":
    config = BiralesConfig(['/home/lessju/Software/birales/pybirales/configuration/birales.ini',
                            '/home/lessju/Software/birales/pybirales/configuration/generator.ini'])

    # Override some settings
    settings.beamformer.calibrate_subarrays = False

    pipeline = DummyGpuPipeline()
    pipeline.build()
    pipeline.manager.start_pipeline()

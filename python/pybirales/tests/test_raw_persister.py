from pybirales import settings

from pybirales.birales_config import BiralesConfig
from pybirales.pipeline.modules.generators.generator import DummyDataGenerator
from pybirales.pipeline.modules.persisters.raw_persister import RawPersister
from pybirales.pipeline.pipeline import PipelineManagerBuilder


class TestRawPersisterPipeline(PipelineManagerBuilder):
    def __init__(self):
        super().__init__()
        self.manager.name = "Test Raw Persister Pipeline"
        self._id = "test_raw_persister_pipeline_builder"

        # Keep a handle to the raw persister for testing triggering
        self._raw_persister = None

    def build(self):
        """ Build the pipeline """
        # Initialise the modules
        generator = DummyDataGenerator(settings.generator)
        self._raw_persister = RawPersister(settings.raw_persister, generator.output_blob)

        # Add modules to pipeline manager
        self.manager.add_module("generator", generator)
        self.manager.add_module("persister", self._raw_persister)

    def trigger_persister(self, start_time, duration):
        """ Issue a trigger to the persister """
        self._raw_persister.trigger_to_file(start_time, duration)


if __name__ == "__main__":
    # Load configuration
    config = BiralesConfig(['/home/lessju/Software/birales/configuration/birales.ini',
                            '/home/lessju/Software/birales/configuration/generator.ini'])

    # Override some settings
    settings.observation.detection_trigger_enabled = True

    # Create pipeline
    pipeline = TestRawPersisterPipeline()
    pipeline.build()

    pipeline.manager.start_pipeline()

from pybirales import settings
from pybirales.birales_config import BiralesConfig
from pybirales.pipeline.base.pipeline_manager import PipelineManager
from pybirales.pipeline.modules.beamformer.beamformer import Beamformer
from pybirales.pipeline.modules.channeliser import PFB
from pybirales.pipeline.modules.correlator import Correlator
from pybirales.pipeline.modules.detection.filter import Filter
from pybirales.pipeline.modules.detection.preprocessor import PreProcessor
from pybirales.pipeline.modules.persisters.beam_persister import BeamPersister
from pybirales.pipeline.modules.readers.raw_data_reader import RawDataReader
from pybirales.pipeline.modules.terminator import Terminator


obs_config = BiralesConfig(config_file_path="/home/lessju/Software/birales/pybirales/configuration/high_resolution_beamforming.ini")
obs_config.load()

receiver = RawDataReader(settings.rawdatareader)
correlator = Correlator(settings.correlator, receiver.output_blob)
terminator = Terminator(None, correlator.output_blob)

# Add modules to pipeline manager
manager = PipelineManager()
manager.add_module("receiver", receiver)
manager.add_module("correlator", correlator)
manager.add_module("terminator", terminator)

manager.start_pipeline()

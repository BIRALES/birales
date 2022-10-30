from pybirales import settings
from pybirales.birales_config import BiralesConfig
from pybirales.pipeline.base.pipeline_manager import PipelineManager
from pybirales.pipeline.modules.beamformer.beamformer import Beamformer
from pybirales.pipeline.modules.channeliser import PFB
from pybirales.pipeline.modules.persisters.beam_persister import BeamPersister
from pybirales.pipeline.modules.readers.raw_data_reader import RawDataReader
from pybirales.pipeline.modules.terminator import Terminator


obs_config = BiralesConfig(config_file_path="/home/oper/Software/birales/pybirales/configuration/birales.ini")
obs_config.load()

receiver = RawDataReader(settings.rawdatareader)
beamformer = Beamformer(settings.beamformer, receiver.output_blob)
pfb = PFB(settings.channeliser, beamformer.output_blob)
persister = BeamPersister(settings.persister, pfb.output_blob)
terminator = Terminator(None, persister.output_blob)

# Add modules to pipeline manager
manager = PipelineManager()
manager.add_module("receiver", receiver)
manager.add_module("beamformer", beamformer)
manager.add_module("pfb", pfb)
manager.add_module("persister", persister)
manager.add_module("terminator", terminator)

manager.start_pipeline()

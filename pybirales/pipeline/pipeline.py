import abc

from pybirales import settings
from pybirales.pipeline.base.pipeline_manager import PipelineManager
from pybirales.pipeline.modules.beamformer.beamformer import Beamformer
from pybirales.pipeline.modules.channeliser import PFB
from pybirales.pipeline.modules.persisters.corr_matrix_persister import CorrMatrixPersister
from pybirales.pipeline.modules.correlator import Correlator
from pybirales.pipeline.modules.detection.detector import Detector
from pybirales.pipeline.modules.persisters.persister import Persister
from pybirales.pipeline.modules.readers.raw_data_reader import RawDataReader
from pybirales.pipeline.modules.persisters.raw_persister import RawPersister
from pybirales.pipeline.modules.receivers.receiver import Receiver
from pybirales.pipeline.modules.terminator import Terminator


class PipelineManagerBuilder:
    def __init__(self):
        # Initialise the Pipeline Manager
        self.manager = PipelineManager()

    @abc.abstractmethod
    def build(self):
        pass


class DetectionPipelineMangerBuilder(PipelineManagerBuilder):
    def __init__(self):
        PipelineManagerBuilder.__init__(self)

        self.manager.name = 'Detection Pipeline'

    def build(self):
        """
        This script runs the multi-pixel pipeline with debris detection enabled,
        using the specified CONFIGURATION.

        :return:
        """

        if settings.manager.offline:
            receiver = RawDataReader(settings.rawdatareader)
            self.manager.name += ' (Offline)'
        else:
            receiver = Receiver(settings.receiver)

        if settings.manager.save_raw:
            persister_raw = RawPersister(settings.rawpersister, receiver.output_blob)
            beamformer = Beamformer(settings.beamformer, persister_raw.output_blob)
            self.manager.add_module("persister_raw", persister_raw)
        else:
            beamformer = Beamformer(settings.beamformer, receiver.output_blob)
        ppf = PFB(settings.channeliser, beamformer.output_blob)

        if settings.manager.save_beam:
            persister = Persister(settings.persister, ppf.output_blob)
            detector = Detector(settings.detection, persister.output_blob)
            self.manager.add_module("persister", persister)
        else:
            detector = Detector(settings.detection, ppf.output_blob)

        # Add modules to pipeline manager
        self.manager.add_module("receiver", receiver)
        self.manager.add_module("beamformer", beamformer)
        self.manager.add_module("ppf", ppf)
        self.manager.add_module("detector", detector)


class StandAlonePipelineMangerBuilder(PipelineManagerBuilder):
    def __init__(self):
        PipelineManagerBuilder.__init__(self)

        self.manager.name = 'Standalone Pipeline'

    def build(self):
        """
        This script runs the standalone test pipeline,
        using the specified CONFIGURATION.

        """

        reader = RawDataReader(settings.rawdatareader)
        beamformer = Beamformer(settings.beamformer, reader.output_blob)
        pfb = PFB(settings.channeliser, beamformer.output_blob)
        persister = Persister(settings.persister, pfb.output_blob)
        terminator = Terminator(None, persister.output_blob)

        # Add modules to pipeline manager
        self.manager.add_module("reader", reader)
        self.manager.add_module("beamformer", beamformer)
        self.manager.add_module("pfb", pfb)
        self.manager.add_module("persister", persister)
        self.manager.add_module("terminator", terminator)


class TestReceiverPipelineMangerBuilder(PipelineManagerBuilder):
    def __init__(self):
        PipelineManagerBuilder.__init__(self)

        self.manager.name = 'Test Receiver Pipeline'

    def build(self):
        """
        This script runs the test receiver pipeline,
        using the specified CONFIGURATION.
        """

        # Initialise the modules
        receiver = Receiver(settings.receiver)
        terminator = Terminator(None, receiver.output_blob)

        # Add modules to pipeline manager
        self.manager.add_module("receiver", receiver)
        self.manager.add_module("terminator", terminator)


class MultiPipelineMangerBuilder(PipelineManagerBuilder):
    def __init__(self):
        PipelineManagerBuilder.__init__(self)

        self.manager.name = 'Multi Pipeline'

    def build(self):
        """
        This script runs the multi-pixel pipeline together with the correlator pipeline,
        using the specified CONFIGURATION.
        """
        # Initialise the modules
        receiver = Receiver(settings.receiver)

        beamformer = Beamformer(settings.beamformer, receiver.output_blob)
        ppf = PFB(settings.channeliser, beamformer.output_blob)
        persister = Persister(settings.persister, ppf.output_blob)

        correlator = Correlator(settings.correlator, receiver.output_blob)
        corr_persister = CorrMatrixPersister(settings.corrmatrixpersister, correlator.output_blob)

        # Add modules to pipeline manager
        self.manager.add_module("receiver", receiver)
        self.manager.add_module("beamformer", beamformer)
        self.manager.add_module("ppf", ppf)
        self.manager.add_module("persister", persister)
        self.manager.add_module("correlator", correlator)
        self.manager.add_module("corr_persister", corr_persister)


class CorrelatorPipelineManagerBuilder(PipelineManagerBuilder):
    def __init__(self):
        PipelineManagerBuilder.__init__(self)

        self.manager.name = 'Correlator Pipeline'

    def build(self):
        if settings.manager.offline:
            receiver = RawDataReader(settings.rawdatareader)
            self.manager.name += ' (Offline)'
        else:
            receiver = Receiver(settings.receiver)

        if settings.manager.save_raw:
            persister_raw = RawPersister(settings.rawpersister, receiver.output_blob)
            correlator = Correlator(settings.correlator, persister_raw.output_blob)
            self.manager.add_module("persister_raw", persister_raw)
        else:
            correlator = Correlator(settings.correlator, receiver.output_blob)

        persister = CorrMatrixPersister(settings.corrmatrixpersister, correlator.output_blob)

        # Add modules to pipeline manager
        self.manager.add_module("receiver", receiver)
        self.manager.add_module("correlator", correlator)
        self.manager.add_module("persister", persister)


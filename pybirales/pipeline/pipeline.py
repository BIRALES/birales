import abc

from pybirales import settings
from pybirales.pipeline.base.pipeline_manager import PipelineManager
from pybirales.pipeline.base.definitions import PipelineBuilderIsNotAvailableException
from pybirales.pipeline.modules.beamformer.beamformer import Beamformer
from pybirales.pipeline.modules.channeliser import PFB
from pybirales.pipeline.modules.correlator import Correlator
from pybirales.pipeline.modules.detection.detector import Detector
from pybirales.pipeline.modules.persisters.corr_matrix_persister import CorrMatrixPersister
from pybirales.pipeline.modules.persisters.persister import Persister
from pybirales.pipeline.modules.persisters.raw_persister import RawPersister
from pybirales.pipeline.modules.readers.raw_data_reader import RawDataReader
from pybirales.pipeline.modules.receivers.receiver import Receiver
from pybirales.pipeline.modules.terminator import Terminator

AVAILABLE_PIPELINES_BUILDERS = ['detection_pipeline',
                                'correlation_pipeline',
                                'standalone_pipeline']


def get_builder_by_id(builder_id):
    """
    Return the pipeline manager builder by name

    @todo - this can converted into a util class an be accessible by any module
    :param builder_id: The name of the pipeline builder
    :raises PipelineBuilderIsNotAvailableException: The pipeline builder does not exist

    :return:
    """

    if builder_id not in AVAILABLE_PIPELINES_BUILDERS:
        raise PipelineBuilderIsNotAvailableException(builder_id, AVAILABLE_PIPELINES_BUILDERS)

    if builder_id == 'detection_pipeline':
        return DetectionPipelineMangerBuilder()
    elif builder_id == 'correlation_pipeline':
        return CorrelatorPipelineManagerBuilder()
    elif builder_id == 'standalone_pipeline':
        return StandAlonePipelineMangerBuilder()


class PipelineManagerBuilder:
    def __init__(self):
        # Initialise the Pipeline Manager
        self.manager = PipelineManager()

        self._id = 'pipeline_manager_builder'

    @abc.abstractmethod
    def build(self):
        pass

    @property
    def id(self):
        return self._id


class DetectionPipelineMangerBuilder(PipelineManagerBuilder):
    def __init__(self):
        PipelineManagerBuilder.__init__(self)

        self.manager.name = 'Detection Pipeline'

        self._id = 'detection_pipeline_builder'

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

        # Saving raw data options
        if settings.manager.save_raw:
            persister_raw = RawPersister(settings.rawpersister, receiver.output_blob)
            beamformer = Beamformer(settings.beamformer, persister_raw.output_blob)
            self.manager.add_module("persister_raw", persister_raw)
        else:
            beamformer = Beamformer(settings.beamformer, receiver.output_blob)

        # Channeliser
        ppf = PFB(settings.channeliser, beamformer.output_blob)

        # Persisting beam and detector options
        if settings.manager.detector_enabled and settings.manager.save_beam:
            persister = Persister(settings.persister, ppf.output_blob)
            detector = Detector(settings.detection, persister.output_blob)
            self.manager.add_module("detector", detector)
            self.manager.add_module("persister", persister)
        elif settings.manager.detector_enabled and not settings.manager.save_beam:
            detector = Detector(settings.detection, ppf.output_blob)
            self.manager.add_module("detector", detector)
        elif not settings.manager.detector_enabled and settings.manager.save_beam:
            persister = Persister(settings.persister, ppf.output_blob)
            terminator = Terminator(settings.terminator, persister.output_blob)
            self.manager.add_module("persister", persister)
            self.manager.add_module("terminator", terminator)
        else:
            terminator = Terminator(settings.terminator, ppf.output_blob)
            self.manager.add_module("terminator", terminator)

        # Add modules to pipeline manager
        self.manager.add_module("receiver", receiver)
        self.manager.add_module("beamformer", beamformer)
        self.manager.add_module("ppf", ppf)


class StandAlonePipelineMangerBuilder(PipelineManagerBuilder):
    def __init__(self):
        PipelineManagerBuilder.__init__(self)

        self.manager.name = 'Standalone Pipeline'

        self._id = 'standalone_pipeline_builder'

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

        self._id = 'test_receiver_pipeline_builder'

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


class CorrelatorPipelineManagerBuilder(PipelineManagerBuilder):
    def __init__(self):
        PipelineManagerBuilder.__init__(self)

        self.manager.name = 'Correlator Pipeline'

        self._id = 'correlator_pipeline_builder'

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

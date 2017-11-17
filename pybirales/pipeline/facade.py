from pybirales.backend.base import settings
from pybirales.backend.base.pipeline_manager import PipelineManager
from pybirales.backend.modules.beamformer.beamformer import Beamformer
from pybirales.backend.modules.channeliser import PFB
from pybirales.backend.modules.persisters.corr_matrix_persister import CorrMatrixPersister
from pybirales.backend.modules.correlator import Correlator
from pybirales.backend.modules.detection.detector import Detector
from pybirales.backend.modules.persisters.persister import Persister
from pybirales.backend.modules.readers.raw_data_reader import RawDataReader
from pybirales.backend.modules.persisters.raw_persister import RawPersister
from pybirales.backend.modules.receivers.receiver import Receiver
from pybirales.backend.modules.terminator import Terminator
from pybirales.backend.plotters.channel_plotter import ChannelisedDataPlotter


class PipelineFacade:

    def __init__(self):
        pass

    @staticmethod
    def detection_pipeline(configuration, debug, save_raw, save_beam):
        """
        This script runs the multi-pixel pipeline with debris detection enabled,
        using the specified CONFIGURATION.

        """

        # Initialise the Pipeline Manager
        manager = PipelineManager(configuration, debug)

        # Initialise the modules
        receiver = Receiver(settings.receiver)

        if save_raw:
            persister_raw = RawPersister(settings.rawpersister, receiver.output_blob)
            beamformer = Beamformer(settings.beamformer, persister_raw.output_blob)
            manager.add_module("persister_raw", persister_raw)
        else:
            beamformer = Beamformer(settings.beamformer, receiver.output_blob)

        ppf = PFB(settings.channeliser, beamformer.output_blob)

        if save_beam:
            persister = Persister(settings.persister, ppf.output_blob)
            detector = Detector(settings.detection, persister.output_blob)
            manager.add_module("persister", persister)
        else:
            detector = Detector(settings.detection, ppf.output_blob)

        # Add modules to pipeline manager
        manager.add_module("receiver", receiver)
        manager.add_module("beamformer", beamformer)
        manager.add_module("ppf", ppf)
        manager.add_module("detector", detector)

        #    manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, ppf.output_blob)
        #    manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)

        manager.start_pipeline()

    @staticmethod
    def standalone_test_pipeline(configuration, debug):
        """
        This script runs the standalone test pipeline,
        using the specified CONFIGURATION.

        """

        # Initialise the Pipeline Manager
        manager = PipelineManager(configuration, debug)

        reader = RawDataReader(settings.rawdatareader)
        beamformer = Beamformer(settings.beamformer, reader.output_blob)
        pfb = PFB(settings.channeliser, beamformer.output_blob)
        persister = Persister(settings.persister, pfb.output_blob)
        terminator = Terminator(None, persister.output_blob)

        # Add modules to pipeline manager
        manager.add_module("reader", reader)
        manager.add_module("beamformer", beamformer)
        manager.add_module("pfb", pfb)
        manager.add_module("persister", persister)
        manager.add_module("terminator", terminator)

        # manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, pfb.output_blob)
        # manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)

        manager.start_pipeline()

    @staticmethod
    def test_receiver_pipeline(configuration, debug):
        """
        This script runs the test receiver pipeline,
        using the specified CONFIGURATION.
        """

        # Initialise the Pipeline Manager
        manager = PipelineManager(configuration, debug)

        # Initialise the modules
        receiver = Receiver(settings.receiver)
        terminator = Terminator(None, receiver.output_blob)

        # Add modules to pipeline manager
        manager.add_module("receiver", receiver)
        manager.add_module("terminator", terminator)

        #  manager.add_plotter("antenna_plotter", AntennaPlotter, settings.antennaplotter, receiver.output_blob)
        #  manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)
        #  manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, ppf.output_blob)

        manager.start_pipeline()

    @staticmethod
    def birales_pipeline(configuration, debug, save_raw):
        """
        This script runs the multipixel pipeline without detection,
        using the specified CONFIGURATION.
        """

        # Initialise the Pipeline Manager
        manager = PipelineManager(configuration, debug)

        # Generate processing modules and data blobs
        receiver = Receiver(settings.receiver)

        if save_raw:
            persister_raw = RawPersister(settings.rawpersister, receiver.output_blob)
            beamformer = Beamformer(settings.beamformer, persister_raw.output_blob)
            manager.add_module("persister_raw", persister_raw)
        else:
            beamformer = Beamformer(settings.beamformer, receiver.output_blob)

        ppf = PFB(settings.channeliser, beamformer.output_blob)
        persister = Persister(settings.persister, ppf.output_blob)
        terminator = Terminator(settings.terminator, persister.output_blob)

        # Add modules to pipeline manager
        manager.add_module("receiver", receiver)
        manager.add_module("beamformer", beamformer)
        manager.add_module("ppf", ppf)
        manager.add_module("persister", persister)
        manager.add_module("terminator", terminator)

        # Add plotters
        # manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)
        # manager.add_plotter("antenna_plotter", AntennaPlotter, settings.antennaplotter, receiver.output_blob)
        # manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, ppf.output_blob)

        manager.start_pipeline()

    @staticmethod
    def multi_pipeline(configuration, debug):
        """
        This script runs the multi-pixel pipeline together with the correlator pipeline,
        using the specified CONFIGURATION.
        """

        # Initialise the Pipeline Manager
        manager = PipelineManager(configuration, debug)

        # Initialise the modules
        receiver = Receiver(settings.receiver)

        beamformer = Beamformer(settings.beamformer, receiver.output_blob)
        ppf = PFB(settings.channeliser, beamformer.output_blob)
        persister = Persister(settings.persister, ppf.output_blob)

        correlator = Correlator(settings.correlator, receiver.output_blob)
        corr_persister = CorrMatrixPersister(settings.corrmatrixpersister, correlator.output_blob)

        # Add modules to pipeline manager
        manager.add_module("receiver", receiver)
        manager.add_module("beamformer", beamformer)
        manager.add_module("ppf", ppf)
        manager.add_module("persister", persister)
        manager.add_module("correlator", correlator)
        manager.add_module("corr_persister", corr_persister)

        manager.start_pipeline()

    @staticmethod
    def save_raw_data_pipeline(configuration, debug):
        """
         This script runs save the incoming raw data to file,
         using the specified CONFIGURATION.
        """

        # Initialise the Pipeline Manager
        manager = PipelineManager(configuration, debug)

        # Initialise the modules
        receiver = Receiver(settings.receiver)
        persister_raw = RawPersister(settings.rawpersister, receiver.output_blob)

        # Add modules to pipeline manager
        manager.add_module("receiver", receiver)
        manager.add_module("persister_raw`", persister_raw)

        manager.start_pipeline()

    @staticmethod
    def offline_birales_pipeline(configuration, debug):
        """
         This script runs the offline (read from file) multi-pixel pipeline with detection,
         using the specified CONFIGURATION.
        """

        # Initialise the Pipeline Manager
        manager = PipelineManager(configuration, debug)

        # Initialise the modules
        reader = RawDataReader(settings.rawdatareader)
        # generator = DummyDataGenerator(settings.generator)
        beamformer = Beamformer(settings.beamformer, reader.output_blob)
        ppf = PFB(settings.channeliser, beamformer.output_blob)
        detector = Detector(settings.detection, ppf.output_blob)

        # Add modules to pipeline manager
        manager.add_module("reader", reader)
        manager.add_module("beamformer", beamformer)
        manager.add_module("ppf", ppf)
        manager.add_module("detector", detector)

        # manager.add_plotter("antenna_plotter", AntennaPlotter, settings.antennaplotter, reader.output_blob)
        # manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)
        manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, ppf.output_blob)

        manager.start_pipeline()

    @staticmethod
    def offline_correlator_pipeline(configuration, debug):
        """
         This script runs the offline (read from file) correlator pipeline,
         using the specified CONFIGURATION.
        """

        # Initialise the Pipeline Manager
        manager = PipelineManager(configuration, debug)

        # Initialise the modules
        reader = RawDataReader(settings.rawdatareader)
        correlator = Correlator(settings.correlator, reader.output_blob)
        persister = CorrMatrixPersister(settings.corrmatrixpersister, correlator.output_blob)

        # Add modules to pipeline manager
        manager.add_module("reader", reader)
        manager.add_module("correlator", correlator)
        manager.add_module("persister", persister)

        manager.start_pipeline()

    @staticmethod
    def correlator_pipeline(configuration, debug, save_raw):
        """
        This script runs the correlator pipeline,
        using the specified CONFIGURATION.
        """

        # Initialise the Pipeline Manager
        manager = PipelineManager(configuration, debug)

        # Generate processing modules and data blobs
        receiver = Receiver(settings.receiver)

        if save_raw:
            persister_raw = RawPersister(settings.rawpersister, receiver.output_blob)
            correlator = Correlator(settings.correlator, persister_raw.output_blob)
            manager.add_module("persister_raw", persister_raw)
        else:
            correlator = Correlator(settings.correlator, receiver.output_blob)

        persister = CorrMatrixPersister(settings.corrmatrixpersister, correlator.output_blob)

        # Add modules to pipeline manager
        manager.add_module("receiver", receiver)
        manager.add_module("correlator", correlator)
        manager.add_module("persister", persister)

        manager.start_pipeline()

#!/usr/bin/python
import click

from pybirales.modules.corr_matrix_persister import CorrMatrixPersister
from pybirales.modules.correlator import Correlator
from pybirales.modules.terminator import Terminator
from pybirales.base.pipeline_manager import PipelineManager
from pybirales.base import settings
from pybirales.modules.raw_data_reader import RawDataReader
from pybirales.modules.beamformer import Beamformer
from pybirales.modules.channeliser import PFB
from pybirales.modules.generator import DummyDataGenerator
from pybirales.modules.persister import Persister
from pybirales.modules.raw_persister import RawPersister
from pybirales.modules.receiver import Receiver
from pybirales.modules.detector import Detector
from pybirales.plotters.channel_plotter import ChannelisedDataPlotter
from pybirales.plotters.antenna_plotter import AntennaPlotter
from pybirales.plotters.bandpass_plotter import BandpassPlotter
from pybirales.modules.monitoring.server import app


@click.group()
def cli():
    pass


@cli.command()
@click.argument('configuration')
@click.option('--debug/--no-debug', default=False, help='Specify whether (or not) you\'d like to log debug messages.')
@click.option('--save-raw/--no-save-raw', default=False, help='Save raw data')
@click.option('--save-beam/--no-save-beam', default=False, help='Save beam data')
def detection_pipeline(configuration, debug, save_raw, save_beam):
    """
    This script runs the BIRALES pipeline with post processing enabled,
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


@cli.command()
@click.argument('configuration', default='config/birales.ini')
@click.option('--debug/--no-debug', default=False, help='Specify whether (or not) you\'d like to log debug messages.')
def standalone_test(configuration, debug):
    """
     This script runs the standalone test pipeline,
     using the specified CONFIGURATION.
    """

    # Initialise the Pipeline Manager
    manager = PipelineManager(configuration, debug)

    # Initialise the modules
    generator = DummyDataGenerator(settings.generator)
    beamformer = Beamformer(settings.beamformer, generator.output_blob)
    pfb = PFB(settings.channeliser, beamformer.output_blob)
    terminator = Terminator(settings.terminator, pfb.output_blob)

    # Add modules to pipeline manager
    manager.add_module("generator", generator)
    manager.add_module("beamformer", beamformer)
    manager.add_module("pfb", pfb)
    manager.add_module("terminator", terminator)

#    manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, pfb.output_blob)
    manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)

    manager.start_pipeline()


@cli.command()
@click.argument('configuration', default='config/birales.ini')
@click.option('--debug/--no-debug', default=False, help='Specify whether (or not) you\'d like to log debug messages.')
def test_receiver(configuration, debug):
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

    manager.add_plotter("antenna_plotter", AntennaPlotter, settings.antennaplotter, receiver.output_blob)
  #  manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)
  #  manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, ppf.output_blob)

    manager.start_pipeline()


@cli.command()
@click.argument('configuration', default='config/birales.ini')
@click.option('--debug/--no-debug', default=False, help='Specify whether (or not) you\'d like to log debug messages.')
@click.option('--save-raw/--no-save-raw', default=False, help='Save raw data?')
def birales_pipeline(configuration, debug, save_raw):
    """
    This script runs the default BIRALES pipeline,
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
    manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)
    # manager.add_plotter("antenna_plotter", AntennaPlotter, settings.antennaplotter, receiver.output_blob)
    # manager.add_plotter("channel_plotter", ChannelisedDataPlotter, settings.channelplotter, ppf.output_blob)

    manager.start_pipeline()

@cli.command()
@click.argument('configuration', default='config/birales.ini')
@click.option('--debug/--no-debug', default=False, help='Specify whether (or not) you\'d like to log debug messages.')
def correlator_pipeline(configuration, debug):
    """
    This script runs the correlator test pipeline,
    using the specified CONFIGURATION.
    """

    # Initialise the Pipeline Manager
    manager = PipelineManager(configuration, debug)

    # Generate processing modules and data blobs
    receiver = Receiver(settings.receiver)
    correlator = Correlator(settings.correlator, receiver.output_blob)
    persister = CorrMatrixPersister(settings.corrmatrixpersister, correlator.output_blob)

    # Add modules to pipeline manager
    manager.add_module("receiver", receiver)
    manager.add_module("correlator", correlator)
    manager.add_module("persister", persister)

    manager.start_pipeline()


@cli.command()
@click.argument('configuration', default='config/birales.ini')
@click.option('--debug/--no-debug', default=False, help='Specify whether (or not) you\'d like to log debug messages.')
def multipipeline(configuration, debug):
    """
    This script runs the test receiver pipeline,
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

@cli.command()
@click.argument('configuration', default='config/birales.ini')
@click.option('--debug/--no-debug', default=False, help='Specify whether (or not) you\'d like to log debug messages.')
def save_raw_data(configuration, debug):
    """
     This script runs the standalone test pipeline,
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

@cli.command()
@click.argument('configuration', default='config/birales.ini')
@click.option('--debug/--no-debug', default=False, help='Specify whether (or not) you\'d like to log debug messages.')
def offline_birales_pipeline(configuration, debug):
    """
     This script runs the standalone test pipeline,
     using the specified CONFIGURATION.
    """

    # Initialise the Pipeline Manager
    manager = PipelineManager(configuration, debug)

    # Initialise the modules
    reader = RawDataReader(settings.rawdatareader)
    beamformer = Beamformer(settings.beamformer, reader.output_blob)
    ppf = PFB(settings.channeliser, beamformer.output_blob)
    detector = Detector(settings.detection, ppf.output_blob)
    terminator = Terminator(settings.terminator, ppf.output_blob)

    # Add modules to pipeline manager
    manager.add_module("reader", reader)
    manager.add_module("ppf", ppf)
    manager.add_module("beamformer", beamformer)
    manager.add_module("detector", detector)

    # manager.add_plotter("antenna_plotter", AntennaPlotter, settings.antennaplotter, reader.output_blob)
    # manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)

    manager.start_pipeline()


@cli.command()
@click.argument('configuration', default='config/birales.ini')
@click.option('--debug/--no-debug', default=False, help='Specify whether (or not) you\'d like to log debug messages.')
def offline_correlator(configuration, debug):
    """
     This script runs the standalone test pipeline,
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

@cli.command()
@click.option('--port', default=5000)
@click.option('--debug/--no-debug', default=False, help='Specify whether (or not) you\'d like to log debug messages.')
def start_server(port, debug):
    app.config['SECRET_KEY'] = 'secret!'
    app.config['DEBUG'] = debug
    app.run(host='0.0.0.0', debug=debug, port=port)


if __name__ == "__main__":
    cli()

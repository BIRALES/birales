#!/usr/bin/python
import click

from pybirales.modules.terminator import Terminator
from pybirales.base.pipeline_manager import PipelineManager
from pybirales.base import settings
from pybirales.modules.beamformer import Beamformer
from pybirales.modules.channeliser import PFB
from pybirales.modules.generator import DummyDataGenerator
from pybirales.modules.persister import Persister
from pybirales.modules.receiver import Receiver
from pybirales.modules.detector import Detector
from pybirales.plotters.bandpass_plotter import BandpassPlotter
from pybirales.plotters.antenna_plotter import AntennaPlotter


@click.group()
def cli():
    pass


@cli.command()
@click.argument('configuration', default='config/birales.ini')
@click.option('--debug/--no-debug', default=True)
def birales_pipeline_with_post_processing(configuration, debug):
    """
    The BIRALES pipeline with post processing enabled
    @todo this will eventually be integrated to the default BIRALES pipeline

    :param configuration: The configuration file
    :param debug: If specified, debug messages will be logged
    :return:
    """

    # Initialise the Pipeline Manager
    manager = PipelineManager(configuration, debug)

    # Initialise the modules
    receiver = DummyDataGenerator(settings.generator)
    beamformer = Beamformer(settings.beamformer, receiver.output_blob)
    ppf = PFB(settings.channeliser, beamformer.output_blob)
    detector = Detector(settings.detection, ppf.output_blob)

    # Add modules to pipeline manager
    manager.add_module("receiver", receiver)
    manager.add_module("beamformer", beamformer)
    manager.add_module("ppf", ppf)
    manager.add_module("detector", detector)

    manager.start_pipeline()


@cli.command()
@click.argument('configuration', default='config/birales.ini')
@click.option('--debug/--no-debug', default=True)
def standalone_test(configuration, debug):
    """
    The standalone_test pipeline

    :param configuration: The configuration file
    :param debug: If specified, debug messages will be logged
    :return:
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


@cli.command()
@click.argument('configuration', default='config/birales.ini')
@click.option('--debug/--no-debug', default=True)
def test_receiver(configuration, debug):
    """
    The test receiver pipeline

    :param configuration: The configuration file
    :param debug: If specified, debug messages will be logged
    :return:
    """

    # Initialise the Pipeline Manager
    manager = PipelineManager(configuration, debug)

    # Initialise the modules
    receiver = Receiver(settings.receiver)
    ppf = PFB(settings.channeliser, receiver.output_blob)
    terminator = Terminator(None, ppf.output_blob)

    # Add modules to pipeline manager
    manager.add_module("receiver", receiver)
    manager.add_module("ppf", ppf)
    manager.add_module("terminator", terminator)

    manager.add_plotter("antenna_plotter", AntennaPlotter, settings.antennaplotter, receiver.output_blob)


@cli.command()
@click.argument('configuration', default='config/birales.ini')
@click.option('--debug/--no-debug', default=True)
def birales_pipeline(configuration, debug):
    """
    The default BIRALES pipeline

    :param configuration: The configuration file
    :param debug: If specified, debug messages will be logged
    :return:
    """
    # Initialise the Pipeline Manager
    manager = PipelineManager(configuration, debug)

    # Generate processing modules and data blobs
    receiver = Receiver(settings.receiver)
    beamformer = Beamformer(settings.beamformer, receiver.output_blob)
    ppf = PFB(settings.channeliser, beamformer.output_blob)
    persister = Persister(settings.persister, ppf.output_blob)

    # Add modules to pipeline manager
    manager.add_module("receiver", receiver)
    manager.add_module("beamformer", beamformer)
    manager.add_module("ppf", ppf)
    manager.add_module("persister", persister)

    # Add plotters
    manager.add_plotter("bandpass_plotter", BandpassPlotter, settings.bandpassplotter, ppf.output_blob)


if __name__ == "__main__":
    cli()

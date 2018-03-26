import datetime

import click

from pybirales.birales import BiralesFacade
from pybirales.birales_config import BiralesConfig
from pybirales.cli.helpers import update_config
from pybirales.pipeline.pipeline import DetectionPipelineMangerBuilder, CorrelatorPipelineManagerBuilder, \
    StandAlonePipelineMangerBuilder, TestReceiverPipelineMangerBuilder


@click.group()
@click.option('--name', '-n', 'name', help='The name of the observation')
@click.option('--debug/--no-debug', default=False)
@click.option('--duration', 'duration', default=None, help='The duration of the observation (0 to run indefinitely)')
@click.pass_context
def pipelines(ctx, name, debug, duration):
    if not name:
        name = 'Observation_{:%Y-%m-%dT%H%M}'.format(datetime.datetime.utcnow())

    ctx.obj = {
        'observation': {
            'name': name,
            'duration': duration
        },
        'manager': {
            'debug': debug
        }
    }

    ctx.obj = update_config(ctx.obj, 'observation', 'name', name)
    ctx.obj = update_config(ctx.obj, 'manager', 'debug', debug)


@pipelines.command(short_help='Run the Detection Pipeline')
@click.option('--config', '-c', 'config_file_path', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.option('--tx', 'tx', help='The transmission frequency in MHz')
@click.option('--pointing', 'pointing', default=12.4, help='Reference Declination of the Beam Former')
@click.pass_context
def detection_pipeline(ctx, config_file_path, tx, pointing):
    """

    Run the Detection Pipeline

    :param ctx:
    :param tx:
    :param pointing:
    :param config_file_path: The default configuration file to be used.
    :return:
    """

    # if tx:
    ctx.obj = update_config(ctx.obj, 'observation', 'transmitter_frequency', tx)

    # if pointing:
    ctx.obj = update_config(ctx.obj, 'beamformer', 'reference_pointing', pointing)

    # Load the BIRALES configuration from file
    config = BiralesConfig(config_file_path, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Build the Pipeline Manager using the Detection Pipeline Manager Builder
    manager = bf.build_pipeline(DetectionPipelineMangerBuilder())

    # Finally, start the observation
    bf.start_observation(pipeline_manager=manager)


@pipelines.command(short_help='Run the Correlation Pipeline')
@click.option('--config', '-c', 'config_file_path', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.pass_context
def correlation_pipeline(ctx, config_file_path):
    """
    Run the Correlation Pipeline

    :param ctx:
    :param config_file_path: The default configuration file to be used.
    :return:
    """

    # Load the BIRALES configuration from file
    config = BiralesConfig(config_file_path, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Build the Pipeline Manager using the Correlator Pipeline Manager Builder
    manager = bf.build_pipeline(CorrelatorPipelineManagerBuilder())

    # Finally, start the observation
    bf.start_observation(pipeline_manager=manager)


@pipelines.command(short_help='Run the stand alone Pipeline')
@click.option('--config', '-c', 'config_file_path', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
def standalone_pipeline(config_file_path):
    """
    Run the Stand Alone Pipeline

    :param config_file_path: The default configuration file to be used.
    :return:
    """

    # Load the BIRALES configuration from file
    config = BiralesConfig(config_file_path)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Build the Pipeline Manager using the Stand Alone Pipeline Manager Builder
    manager = bf.build_pipeline(StandAlonePipelineMangerBuilder())

    # Finally, start the observation
    bf.start_observation(pipeline_manager=manager)


@pipelines.command(short_help='Run the stand alone Pipeline')
@click.option('--config', '-c', 'config_file_path', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
def test_receiver_pipeline(config_file_path):
    """
    Run the Stand Alone Pipeline

    :param config_file_path: The default configuration file to be used.
    :return:
    """

    # Load the BIRALES configuration from file
    config = BiralesConfig(config_file_path)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Build the Pipeline Manager using the Stand Alone Pipeline Manager Builder
    manager = bf.build_pipeline(TestReceiverPipelineMangerBuilder())

    # Finally, start the observation
    bf.start_observation(pipeline_manager=manager)

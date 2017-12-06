import click
from pybirales.birales import BiralesFacade, BiralesConfig
from pybirales.pipeline.pipeline import DetectionPipelineMangerBuilder, CorrelatorPipelineManagerBuilder
from pybirales.cli.helpers import update_config


@click.group()
@click.option('--name', '-n', 'name', default='observation', help='The name of the observation')
@click.option('--debug/--no-debug', default=False)
@click.option('--duration', 'duration', default=0, help='The duration of the observation (0 to run indefinitely)')
@click.pass_context
def pipelines(ctx, name, debug, duration):
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
@click.argument('configuration', type=click.Path(exists=True))
@click.option('--tx', 'tx', default=410.07, help='The transmission frequency in MHz')
@click.option('--pointing', 'pointing', default=12.3, help='Reference Declination of the Beam Former')
@click.pass_context
def detection_pipeline(ctx, configuration, tx, pointing):
    """

    Run the Detection Pipeline

    :param ctx:
    :param tx:
    :param pointing:
    :param configuration: The default configuration file to be used.
    :return:
    """

    ctx.obj = update_config(ctx.obj, 'observation', 'transmitter_frequency', tx)
    ctx.obj = update_config(ctx.obj, 'beamformer', 'reference_pointing', pointing)

    # Load the BIRALES configuration from file
    config = BiralesConfig(configuration, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Build the Pipeline Manager using the Detection Pipeline Manager Builder
    manager = bf.build_pipeline(DetectionPipelineMangerBuilder())

    # Finally, start the observation
    bf.start_observation(pipeline_manager=manager)


@pipelines.command(short_help='Run the Correlation Pipeline')
@click.argument('configuration', type=click.Path(exists=True))
def correlation_pipeline(configuration):
    """
    Run the Correlation Pipeline

    :param configuration: The default configuration file to be used.
    :return:
    """

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration)

    # Build the Pipeline Manager using the Correlator Pipeline Manager Builder
    manager = bf.build_pipeline(CorrelatorPipelineManagerBuilder())

    # Finally, start the observation
    bf.start_observation(pipeline_manager=manager)

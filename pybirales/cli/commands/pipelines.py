import click
from pybirales.birales import BiralesFacade, BiralesConfig
from pybirales.pipeline.pipeline import DetectionPipelineMangerBuilder, CorrelatorPipelineManagerBuilder


@click.group()
@click.option('--option1')
def pipelines(option1):
    pass


def parse_options(ctx):
    print(ctx.args)
    for option in ctx.args:
        print(option)

    # print({ctx.args[i][2:]: ctx.args[i + 1] for i in range(0, len(ctx.args))})

    return {}


@pipelines.command(short_help='Run the Detection Pipeline', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument('configuration', type=click.Path(exists=True))
@click.pass_context
def detection_pipeline(ctx, configuration):
    """
    Run the Detection Pipeline

    :param ctx:
    :param configuration: The default configuration file to be used.
    :return:
    """

    # Load the BIRALES configuration from file
    config = BiralesConfig(configuration, parse_options(ctx))

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Build the Pipeline Manager using the Detection Pipeline Manager Builder
    manager = bf.build_pipeline(DetectionPipelineMangerBuilder())

    # Finally, start the Pipeline
    manager.start_pipeline()


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

    # Finally, start the Pipeline
    manager.start_pipeline()

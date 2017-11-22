import click
from pybirales.birales import BiralesFacade
from pybirales.pipeline.pipeline import DetectionPipelineMangerBuilder, CorrelatorPipelineManagerBuilder


@click.group()
def pipelines():
    pass


@pipelines.command(short_help='Run the Detection Pipeline')
@click.argument('configuration', type=click.Path(exists=True))
def detection_pipeline(configuration):
    """
    Run the Detection Pipeline

    :param configuration: The default configuration file to be used.
    :return:
    """

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration)

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

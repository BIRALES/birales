import logging

import click
import time

from pybirales.services.instrument.backend import Backend

from pybirales.services.instrument.best2 import BEST2

from pybirales.birales import BiralesFacade, BiralesConfig
from pybirales.pipeline.pipeline import CorrelatorPipelineManagerBuilder


@click.group()
@click.pass_context
def services(ctx):
    pass


@services.command()
@click.argument('configuration', type=click.Path(exists=True), required=True)
@click.pass_context
def calibration(ctx, configuration):
    # Load the BIRALES configuration from file
    config = BiralesConfig(configuration, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Build the Pipeline Manager using the Correlator Pipeline Manager Builder
    manager = bf.build_pipeline(CorrelatorPipelineManagerBuilder())

    # Calibrate the Instrument
    bf.calibrate(correlator_pipeline_manager=manager)


@services.command()
@click.argument('configuration', type=click.Path(exists=True), required=True)
@click.pass_context
def run_server(ctx, configuration):
    # Load the BIRALES configuration from file
    config = BiralesConfig(configuration, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Start the Flask server
    bf.start_server()


@services.command()
@click.argument('configuration', type=click.Path(exists=True), required=True)
@click.pass_context
def init_roach(ctx, configuration):
    # Load the BIRALES configuration from file
    config = BiralesConfig(configuration, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    BiralesFacade(configuration=config)

    # Initialise the roach
    backend = Backend.Instance()
    time.sleep(2)
    backend.start()


@services.command()
@click.argument('configuration', type=click.Path(exists=True), required=True)
@click.option('--pointing', default=-90, help='Where to point BEST-II [Default: no pointing]')
@click.pass_context
def best_pointing(ctx, configuration, pointing):
    # Load the BIRALES configuration from file
    config = BiralesConfig(configuration, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    time.sleep(1)

    # Get BEST-II instance
    best2 = BEST2.Instance()

    time.sleep(1)

    # Point BEST-II or get current pointing
    if pointing != -90:
        best2.move_to_declination(pointing)
    else:
        logging.info("BEST-II pointing to {}".format(best2.current_pointing))

    # Clean up server
    best2.stop_best2_server()


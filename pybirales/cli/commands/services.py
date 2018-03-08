import logging
import time

import click

from pybirales.birales import BiralesFacade
from pybirales.birales_config import BiralesConfig
from pybirales.cli.helpers import update_config
from pybirales.pipeline.pipeline import CorrelatorPipelineManagerBuilder
from pybirales.services.instrument.backend import Backend
from pybirales.services.instrument.best2 import BEST2
from pybirales.pipeline.base.definitions import BEST2PointingException


@click.group()
@click.pass_context
def services(ctx):
    pass


@services.command()
@click.option('--config', '-c', 'config_filepath', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.option('--name', '-n', 'name', help='The name of the observation')
@click.pass_context
def calibration(ctx, config_filepath, name):
    if not name:
        name = 'Calibration_Observation'
    ctx.obj = {
        'observation': {
            'name': name
        }
    }

    ctx.obj = update_config(ctx.obj, 'observation', 'name', name)

    # Load the BIRALES configuration from file
    config = BiralesConfig(config_filepath, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Build the Pipeline Manager using the Correlator Pipeline Manager Builder
    manager = bf.build_pipeline(CorrelatorPipelineManagerBuilder())

    # Calibrate the Instrument
    bf.calibrate(correlator_pipeline_manager=manager)

    # Stop the birales system
    bf.stop()


@services.command()
@click.option('--config', '-c', 'config_filepath', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.pass_context
def reset_coefficients(ctx, config_filepath):
    # Load the BIRALES configuration from file
    config = BiralesConfig(config_filepath, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Reset the calibration coefficients
    bf.reset_calibration_coefficients()

    # Stop the birales system
    bf.stop()


@services.command()
@click.option('--config', '-c', 'config_filepath', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.pass_context
def run_server(ctx, config_filepath):
    # Load the BIRALES configuration from file
    config = BiralesConfig(config_filepath, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Start the Flask server
    bf.start_server()


@services.command()
@click.option('--config', '-c', 'config_filepath', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.pass_context
def init_roach(ctx, config_filepath):
    # Load the BIRALES configuration from file
    config = BiralesConfig(config_filepath, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    BiralesFacade(configuration=config)

    # Initialise the roach
    backend = Backend.Instance()
    time.sleep(2)
    backend.start()


@services.command()
@click.option('--config', '-c', 'config_filepath', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.option('--pointing', default=None, type=float, help='Where to point BEST-II')
@click.pass_context
def best_pointing(ctx, config_filepath, pointing):
    # Load the BIRALES configuration from file
    config = BiralesConfig(config_filepath, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    _ = BiralesFacade(configuration=config)

    # Get BEST-II instance
    best2 = BEST2.Instance()

    logging.info("BEST-II current declination is: {:0.2f}".format(best2.current_pointing))

    try:
        if pointing:
            best2.move_to_declination(pointing)
    except BEST2PointingException:
        logging.exception("Could not point the BEST-II")
    finally:
        best2.stop_best2_server()

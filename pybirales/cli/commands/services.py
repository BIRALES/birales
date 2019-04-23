import logging
import time

import click

from pybirales.base.observation_manager import CalibrationObservationManager
from pybirales.birales import BiralesFacade
from pybirales.birales_config import BiralesConfig
from pybirales.cli.helpers import enable_notifications
from pybirales.pipeline.base.definitions import BEST2PointingException
from pybirales.services.calibration.calibration import CalibrationFacade
from pybirales.services.instrument.backend import Backend
from pybirales.services.instrument.best2 import BEST2
from pybirales.services.scheduler.observation import ScheduledCalibrationObservation


@click.group()
@click.pass_context
def services(ctx):
    pass


@services.command()
@click.option('--config', '-c', 'config_filepath', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.option('--name', '-n', 'name', help='The name of the observation')
@click.option('--debug/--no-debug', default=False)
@click.option('--duration', 'duration', default=3600,
              help='The duration of the correlation pipeline (1 hours by default)')
@click.option('--pointing', 'pointing', default=58.8, help='Declination of the BEST array')
@click.option('--corr_matrix', 'corr_matrix_filepath', type=click.Path(exists=True),
              help='The filepath of the correlation matrix')
@click.pass_context
@enable_notifications
def calibration(ctx, config_filepath, name, debug, duration, pointing, corr_matrix_filepath=None):
    if not name:
        name = 'Calibration_Observation'
    ctx.obj = {
        'observation': {
            'name': name,
        },
        'manager': {
            'debug': debug
        },
        'beamformer': {
            'reference_declination': pointing
        },
        'duration': duration
    }

    # Create a new calibration observation
    calibration_obs = ScheduledCalibrationObservation(name=ctx.obj['observation']['name'],
                                                      pipeline_name='correlation_pipeline',
                                                      config_file=config_filepath,
                                                      config_parameters=ctx.obj)
    # Initialise the calibration manager
    om = CalibrationObservationManager()

    om.run(observation=calibration_obs, corr_matrix_filepath=corr_matrix_filepath)


@services.command()
@click.option('--config', '-c', 'config_filepath', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.pass_context
def reset_coefficients(ctx, config_filepath):
    # Load the BIRALES configuration from file
    config = BiralesConfig(config_filepath, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Initialise the roach
    backend = Backend.Instance()

    time.sleep(1)
    backend.start(program_fpga=True, equalize=True, calibrate=False)

    calib_facade = CalibrationFacade()
    backend.load_calibration_coefficients(amplitude=calib_facade.real_reset_coeffs,
                                          phase=calib_facade.imag_reset_coeffs)

    # Stop the birales system
    bf.stop()


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
    bf = BiralesFacade(configuration=config)

    best2 = BEST2.Instance()

    try:
        best2.connect()

        if pointing:
            best2.move_to_declination(pointing)
    except BEST2PointingException:
        logging.warning('BEST2 Server is not available.')
    else:
        logging.info('Successfully connected to BEST antenna server')
        logging.info("BEST-II current declination is: {:0.2f}".format(best2.current_pointing))
    finally:
        best2.stop_best2_server()

    # Stop the birales system
    bf.stop()

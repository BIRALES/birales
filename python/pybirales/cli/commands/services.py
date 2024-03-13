import click

from pybirales.base.observation_manager import CalibrationObservationManager
from pybirales.cli.helpers import enable_notifications
from pybirales.services.scheduler.observation import ScheduledCalibrationObservation


@click.group()
@click.pass_context
def services(ctx):
    pass


@services.command()
@click.option('--config', '-c', 'config_filepath', type=str, required=True,
              help='The BIRALES configuration file', multiple=True)
@click.option('--name', '-n', 'name', help='The name of the observation')
@click.option('--debug/--no-debug', default=False)
@click.option('--duration', 'duration', default=3600,
              help='The duration of the correlation pipeline (1 hours by default)')
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


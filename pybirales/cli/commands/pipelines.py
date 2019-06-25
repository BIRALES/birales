import datetime

import click

from pybirales.base.observation_manager import ObservationManager
from pybirales.cli.helpers import update_config, enable_notifications
from pybirales.services.scheduler.observation import ScheduledObservation


@click.group()
@click.option('--name', '-n', 'name', help='The name of the observation')
@click.option('--debug/--no-debug', default=False)
@click.option('--duration', 'duration', default=3600, help='The duration of the observation (2 hours by default)')
@click.pass_context

def pipelines(ctx, name, debug, duration):
    if not name:
        name = 'Observation_{:%Y-%m-%dT%H%M}'.format(datetime.datetime.utcnow())

    ctx.obj = {
        'observation': {
            'name': name,
        },
        'manager': {
            'debug': debug
        },
        'duration': duration
    }


@pipelines.command(short_help='Run the Detection Pipeline')
@click.option('--config', '-c', 'config_file_path', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.option('--tx', 'tx', type=float, help='The transmission frequency in MHz')
@click.option('--pointing', 'pointing', type=float, help='Reference Declination of the Beam Former')
@click.pass_context
@enable_notifications
def detection_pipeline(ctx, config_file_path, tx, pointing):
    """

    Run the Detection Pipeline

    :param ctx:
    :param tx:
    :param pointing:
    :param config_file_path: The default configuration file to be used.
    :return:
    """

    if tx:
        ctx.obj = update_config(ctx.obj, 'observation', 'transmitter_frequency', tx)

    if pointing:
        ctx.obj = update_config(ctx.obj, 'beamformer', 'reference_declination', pointing)

    ctx.obj['start_time'] = datetime.datetime.utcnow()

    observation = ScheduledObservation(name=ctx.obj['observation']['name'],
                                       pipeline_name='detection_pipeline',
                                       config_file=config_file_path,
                                       config_parameters=ctx.obj)

    om = ObservationManager()

    om.run(observation)



@pipelines.command(short_help='Run the Correlation Pipeline')
@click.option('--config', '-c', 'config_file_path', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.option('--pointing', 'pointing', help='Reference Declination of the Beam Former')
@click.pass_context
@enable_notifications
def correlation_pipeline(ctx, config_file_path, pointing):
    """
    Run the Correlation Pipeline

    :param ctx:
    :param config_file_path: The default configuration file to be used.
    :return:
    """
    if pointing:
        ctx.obj = update_config(ctx.obj, 'beamformer', 'reference_declination', pointing)

    observation = ScheduledObservation(name=ctx.obj['observation']['name'],
                                       pipeline_name='correlation_pipeline',
                                       config_file=config_file_path,
                                       config_parameters=ctx.obj)

    om = ObservationManager()
    om.run(observation)


@pipelines.command(short_help='Run the stand alone Pipeline')
@click.option('--config', '-c', 'config_file_path', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.pass_context
def standalone_pipeline(ctx, config_file_path):
    """
    Run the Stand Alone Pipeline

    :param config_file_path: The default configuration file to be used.
    :return:
    """

    observation = ScheduledObservation(name=ctx.obj['observation']['name'],
                                       pipeline_name='standalone_pipeline',
                                       config_file=config_file_path,
                                       config_parameters=ctx.obj)

    om = ObservationManager()
    om.run(observation)


@pipelines.command(short_help='Run the stand alone Pipeline')
@click.option('--config', '-c', 'config_file_path', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.pass_context
def test_receiver_pipeline(ctx, config_file_path):
    """
    Run the Stand Alone Pipeline

    :param config_file_path: The default configuration file to be used.
    :return:
    """

    observation = ScheduledObservation(name=ctx.obj['observation']['name'],
                                       pipeline_name='test_receiver_pipeline',
                                       config_file=config_file_path,
                                       config_parameters=ctx.obj)

    om = ObservationManager()
    om.run(observation)


@pipelines.command(short_help='Run the dummy data Pipeline')
@click.option('--config', '-c', 'config_file_path', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.pass_context
def dummy_data_pipeline(ctx, config_file_path):
    """
    Run the Stand Alone Pipeline

    :param config_file_path: The default configuration file to be used.
    :return:
    """

    observation = ScheduledObservation(name=ctx.obj['observation']['name'],
                                       pipeline_name='dummy_data_pipeline',
                                       config_file=config_file_path,
                                       config_parameters=ctx.obj)

    om = ObservationManager()
    om.run(observation)



@pipelines.command(short_help='Run the RSO generator Pipeline')
@click.option('--config', '-c', 'config_file_path', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.pass_context
def rso_generator_pipeline(ctx, config_file_path):
    """
    Run the RSO generator Pipeline

    :param config_file_path: The default configuration file to be used.
    :return:
    """

    observation = ScheduledObservation(name=ctx.obj['observation']['name'],
                                       pipeline_name='rso_generator_pipeline',
                                       config_file=config_file_path,
                                       config_parameters=ctx.obj)

    om = ObservationManager()
    om.run(observation)


@pipelines.command(short_help='Run the Data truncator Pipeline')
@click.option('--config', '-c', 'config_file_path', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
@click.pass_context
def data_truncator_pipeline(ctx, config_file_path):
    """
    Run the Data truncator Pipeline

    :param config_file_path: The default configuration file to be used.
    :return:
    """

    observation = ScheduledObservation(name=ctx.obj['observation']['name'],
                                       pipeline_name='raw_data_truncator_pipeline',
                                       config_file=config_file_path,
                                       config_parameters=ctx.obj)

    om = ObservationManager()
    om.run(observation)
import click
import time
import logging as log
from core.pipeline import SpaceDebrisDetectorPipeline
from core.repository import DataSetRepository, BeamDataRepository
from visualization.api import app
from configuration.application import config
import os


@click.group()
@click.option('--multiproc', help='Needed when running in parallel', type=click.STRING)
@click.option('--client', help='Needed when running in parallel', type=click.STRING)
@click.option('--port', help='Needed when running in parallel', type=click.INT)
@click.option('--file', help='Needed when running in parallel', type=click.STRING)
def cli(multiproc, client, port, file):
    log.debug('Using %s for multiproc', multiproc)
    log.debug('Using %s for client', client)
    log.debug('Using %s for port', port)
    log.debug('Using %s for file', file)


def get_data_sets(observation, data_set):
    data_sets = {}
    observations = [observation]
    if observation == '*':
        observations = [x for x in os.listdir(config.get('io', 'DATA_FILE_PATH'))]
        for obs in observations:
            data_sets[obs] = [x for x in os.listdir(config.get('io', 'DATA_FILE_PATH') + '/' + obs)]
        return data_sets

    if data_set == '*':
        for obs in observations:
            data_sets[obs] = [x for x in os.listdir(config.get('io', 'DATA_FILE_PATH') + '/' + obs)]
    else:
        data_sets[observation] = [data_set]
        if observation == '*':
            for obs in observations:
                data_sets[obs] = [x for x in os.listdir(config.get('io', 'DATA_FILE_PATH') + '/' + obs)]

    return data_sets


@cli.command()
@click.option('--observation', help='Observation you want to process', required=True)
@click.option('--data_set', help='The data set you want to process', required=True)
@click.option('--n_beams', help='Number of beams to process', type=click.INT, required=True)
def post_process(observation, data_set, n_beams):
    """
    Post process the beam data and generate space debris candidates

    :param observation  The observation to be processed
    :param data_set     The data_set to be processed
    :param n_beams      The number of beams to process

    :return:
    """

    data_sets = get_data_sets(observation, data_set)

    for observation, data_sets in data_sets.iteritems():
        for data_set in data_sets:
            before = time.time()
            pipeline = SpaceDebrisDetectorPipeline(observation_name=observation, data_set_name=data_set, n_beams=n_beams)
            pipeline.run()

            log.info('Process finished in %s seconds', round(time.time() - before, 3))


@cli.command()
@click.option('--port', help='The port that the server will run on', type=click.INT)
def run_dev_server(port):
    """
    Run the flask server

    :param port The port that the server will run on
    :return:
    """
    app.run_server(port)


@cli.command()
@click.option('--observation', help='Observation you want to delete', required=True)
@click.option('--data_set', help='The data set you want to delete', required=True)
def reset(observation, data_set):
    """
    Drop the database for the selected data set

    :param observation  The observation that the data set belongs to
    :param data_set     The data set to be deleted
    :return:
    """
    # Build the data_set unique id
    data_set_id = observation + '.' + data_set

    # Delete the beam data for this data_set from the database
    beam_data_repository = BeamDataRepository()
    beam_data_repository.destroy(data_set_id)

    # Delete the data_set from the database
    data_set_repository = DataSetRepository()
    data_set_repository.destroy(data_set_id)


if __name__ == '__main__':
    cli()

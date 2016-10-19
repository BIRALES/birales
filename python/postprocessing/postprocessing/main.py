import click
from core.space_debris_detection import SpaceDebrisDetector


@click.command()
@click.option('--observation', prompt='Please specify the observation:', help='Observation you want to process')
@click.option('--data_set', prompt='Please specify the data set:', help='The data set you want to process')
@click.option('--n_beams', prompt='Please specify the number of beams to process:', help='Number of beams to process')
def application(observation, data_set, n_beams):
    """
    Post process the beam data and generate space debris candidates
    :param observation The observation to be processed
    :param data_set The data_set to be processed
    :param n_beams The number of beams to process
    :return:
    """
    odc = SpaceDebrisDetector(observation_name=observation, data_set_name=data_set, n_beams=n_beams)
    odc.run()


if __name__ == '__main__':
    application()

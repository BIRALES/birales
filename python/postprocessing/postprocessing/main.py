import click
from core.space_debris_detection import SpaceDebrisDetector


@click.command()
@click.option('--observation', prompt='Please specify the observation:', help='Observation you want to process')
@click.option('--data_set', prompt='Please specify the data set:', help='The data set you want to process')
def application(observation, data_set):
    """
    Post process the beam data and generate space debris candidates
    :param observation The observation to be processed
    :param data_set The data_set to be processed
    :return:
    """
    odc = SpaceDebrisDetector(observation=observation, data_set=data_set)
    odc.run()


if __name__ == '__main__':
    application()

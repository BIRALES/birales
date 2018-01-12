import click

from pybirales.birales import BiralesFacade, BiralesConfig


@click.command()
@click.option('--schedule', '-s', 'schedule_file_path', type=click.Path(exists=True), required=True,
              help='The scheduler json file')
@click.option('--format', '-f', 'file_format', default='json', help='The format of the schedule file [tdm/json]')
@click.option('--config', '-c', 'config_file_path', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
def scheduler(schedule_file_path, config_file_path, file_format):
    """
    Schedule a series of observations

    :param schedule_file_path: The file path to the schedule
    :param config_file_path: The Birales configuration file
    :param file_format: The format of the schedule file [tdm/json]
    :return:
    """

    # Load the BIRALES configuration from file
    config = BiralesConfig(config_file_path, {})

    # Initialise the BIRALES Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Start the BIRALES scheduler
    bf.start_scheduler(schedule_file_path, file_format)

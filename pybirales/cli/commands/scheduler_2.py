import logging as log

import click
from pybirales.services.scheduler.scheduler import ObservationsScheduler
from pybirales.services.scheduler.exceptions import SchedulerException, NoObservationsQueuedException
from pybirales.birales import BiralesFacade, BiralesConfig


@click.command()
@click.option('--schedule', '-s', 'schedule_file_path', type=click.Path(exists=True), required=True,
              help='The scheduler json file')
@click.option('--format', '-f', 'file_format', default='json', help='The format of the schedule file [tdm/json]')
@click.option('--config', '-c', 'config_file_path', type=click.Path(exists=True), required=True,
              help='The BIRALES configuration file', multiple=True)
def scheduler2(schedule_file_path, config_file_path, file_format):
    """
    Schedule a series of observations

    :param schedule_file_path: The file path to the schedule
    :param config_file_path: The Birales configuration file
    :param file_format: The format of the schedule file [tdm/json]
    :return:
    """

    # Load the BIRALES configuration from file
    config = BiralesConfig(config_file_path, {})

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    try:
        s = ObservationsScheduler()
        s.load_from_file(schedule_file_path, file_format)
        s.start()
    except KeyboardInterrupt:
        log.info('Ctrl-C received. Terminating the scheduler process.')
    except NoObservationsQueuedException:
        log.info('Could not run Scheduler since no valid observations could be scheduled.')
    except SchedulerException:
        log.exception('A fatal Scheduler error has occurred. Terminating the scheduler process.')
    else:
        log.info('Scheduler finished successfully.')

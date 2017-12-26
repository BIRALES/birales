import logging as log
import sys

import click
from pybirales.services.scheduler.scheduler import ObservationsScheduler


@click.command()
@click.option('--schedule_file', '-s', 'schedule_file_path', type=click.Path(exists=True), required=True,
              help='The scheduler json file')
@click.option('--format', '-f', 'file_format', default='json', help='The format of the schedule file [tdm/json]')
def scheduler2(schedule_file_path, file_format):
    """
    Schedule a series of observations

    :param schedule_file_path: The file path to the schedule
    :param file_format: The format of the schedule file [tdm/json]
    :return:
    """

    try:
        s = ObservationsScheduler()
        s.load_from_file(schedule_file_path, file_format)
        s.start()
    except KeyboardInterrupt:
        log.info('Ctrl-C received. Terminating the scheduler process.')
        sys.exit()
    else:
        log.info('Scheduler finished successfully.')

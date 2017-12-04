import click
import datetime
import logging as log
import json
import sched
import sys
import time
from pybirales.birales import BiralesFacade, BiralesConfig
from pybirales.pipeline.pipeline import DetectionPipelineMangerBuilder, CorrelatorPipelineManagerBuilder


def run_pipeline(observation_settings):
    """
    Run the pipeline using the provided observation settings

    :param observation_settings: A dictionary of settings read from the scheduler json config file
    :return:
    """

    # Load the BIRALES configuration from file
    config = BiralesConfig(observation_settings['config_file'], observation_settings['config_options'])

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Build the Pipeline Manager
    if observation_settings['pipeline'] == 'detection_pipeline':
        builder = DetectionPipelineMangerBuilder()
    elif observation_settings['pipeline'] == 'correlation_pipeline':
        builder = CorrelatorPipelineManagerBuilder()
    else:
        raise Exception('Pipeline not implemented')

    manager = bf.build_pipeline(builder)

    manager.start_pipeline(observation_settings['duration'])


@click.group()
@click.option('--schedule', '-s', 'schedule_file', type=click.Path(exists=True), required=True,
              help='The scheduler json file')
def scheduler(schedule_file):
    """
    Schedule a series of observations

    :param schedule_file: The path to the schedule parameters file
    :return:
    """
    scheduled_observations = json.loads(schedule_file)
    # todo -- need to add validation of the config file

    s = sched.scheduler(time.time, time.sleep)
    now = datetime.datetime.fromtimestamp(int(time.time()))
    for observation in scheduled_observations:
        start_time = datetime.datetime.strptime(observation['start_time'], "%Y-%m-%dT%H:%M:%S")

        # Check that start time is valid
        wait_seconds = (start_time - now).total_seconds()
        if wait_seconds < 1:
            log.error("Scheduled start time must be in the future")
            sys.exit()

        log.info("Scheduling {} to run at {}".format(observation['pipeline'], start_time.isoformat()))
        s.enter(delay=wait_seconds, priority=0, action=run_pipeline, argument=observation)

    log.info('Scheduler initialised.')
    s.run()

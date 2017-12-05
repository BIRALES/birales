import click
import datetime
import dateutil.parser
import humanize
import json
import pytz
import sched
import sys
import threading
import time
from pybirales.birales import BiralesFacade, BiralesConfig
from pybirales.pipeline.pipeline import DetectionPipelineMangerBuilder, CorrelatorPipelineManagerBuilder


def message(status, msg):
    """

    :param status:
    :param msg:
    :return:
    """

    now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    click.echo('{:%Y-%m-%d %H:%M:%S} {}: {}'.format(now, status, msg))


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


@click.command()
@click.option('--schedule', '-s', 'schedule_file_path', type=click.Path(exists=True), required=True,
              help='The scheduler json file')
def scheduler(schedule_file_path):
    """
    Schedule a series of observations

    :param schedule_file_path: The path to the schedule parameters file
    :return:
    """

    with open(schedule_file_path) as json_data:
        scheduled_observations = json.load(json_data)
        message('INFO', 'Loaded schedule from {}'.format(schedule_file_path))

    # todo -- need to add validation of the config file

    s = sched.scheduler(time.time, time.sleep)
    now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    for obs_name, observation in scheduled_observations.iteritems():
        start_time = dateutil.parser.parse(observation['config_parameters']['start_time'])

        # Check that start time is valid
        wait_seconds = (start_time - now).total_seconds()
        if wait_seconds < 1:
            message('INFO', "Scheduled start time for {} must be in the future".format(obs_name))
            sys.exit()

        message('INFO', "Observation {}, using the {} is scheduled to run at {:%Y-%m-%d %H:%M:%S}".format(
            obs_name, observation['pipeline'], start_time))
        s.enter(delay=wait_seconds, priority=0, action=run_pipeline, argument=observation)

    message('INFO', 'Scheduler initialised. {} observations queued.'.format(len(scheduled_observations)))

    # Start a thread to run the events
    scheduler_thread = threading.Thread(target=s.run)
    scheduler_thread.daemon = True
    scheduler_thread.start()

    try:
        # Provide feedback to the user on the status of the scheduler
        while not s.empty():
            now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
            for observation in s.queue:
                start_time = dateutil.parser.parse(observation.argument['config_parameters']['start_time'])
                time_remaining = humanize.naturaltime(now - start_time)
                message('INFO', 'The {} is scheduled to start in {}'.format(observation.argument['pipeline'],
                                                                            time_remaining))

            # Do not show the output again for the next N seconds
            time.sleep(10*60)
    except KeyboardInterrupt:
        message('INFO', 'Ctrl-C received. Terminating the scheduler process.')
        sys.exit()

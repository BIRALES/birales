import click
import datetime
import dateutil.parser
import humanize
import json

import os
import pytz
import sched
import sys
import threading
import time

import signal

from pybirales.birales import BiralesFacade, BiralesConfig
from pybirales.pipeline.pipeline import DetectionPipelineMangerBuilder, CorrelatorPipelineManagerBuilder, \
    StandAlonePipelineMangerBuilder

DEFAULT_WAIT_SECONDS = 5
OBS_PADDING = datetime.timedelta(seconds=60)


def message(status, msg):
    """

    :param status:
    :param msg:
    :return:
    """

    now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    click.echo('{:%Y-%m-%d %H:%M:%S} {}: {}'.format(now, status, msg))


def monitoring_thread(s):
    """
    Monitor the status of the scheduler

    :param s: The scheduler instance
    :return:
    """
    while not s.empty():
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        for observation in s.queue:
            obs_settings = observation.argument[0]
            start_time = dateutil.parser.parse(obs_settings['config_parameters']['start_time'])
            time_remaining = humanize.naturaltime(now - start_time)
            message('INFO', 'The {} for the "{}" observation is scheduled to start in {}'.format(
                obs_settings['pipeline'],
                obs_settings['name'],
                time_remaining))

        # Do not show the output again for the next N seconds
        time.sleep(60)


def start_observation(observation_settings):
    """
    Run the pipeline using the provided observation settings

    :param observation_settings: A dictionary of settings read from the scheduler json config file
    :return:
    """
    bf = None

    def _signal_handler(signum, frame):
        """ Capturing interrupt signal """
        message('INFO', "Ctrl-C detected by process %s, stopping pipeline" % os.getpid())

        if bf is not None:
            bf.stop_observation()

    # Set interrupt signal handler
    signal.signal(signal.SIGINT, _signal_handler)

    # Load the BIRALES configuration from file
    config = BiralesConfig(observation_settings['config_file'], observation_settings['config_parameters'])

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Build the Pipeline Manager
    if observation_settings['pipeline'] == 'detection_pipeline':
        builder = DetectionPipelineMangerBuilder()
    elif observation_settings['pipeline'] == 'correlation_pipeline':
        builder = CorrelatorPipelineManagerBuilder()
    elif observation_settings['pipeline'] == 'standalone_pipeline':
        builder = StandAlonePipelineMangerBuilder()
    else:
        raise Exception('Pipeline not implemented')

    manager = bf.build_pipeline(builder)

    # Start observation
    bf.start_observation(pipeline_manager=manager)

def obs_overlaps_schedule(schedule, start_time, end_time):
    """
    Check if this observation overlaps with the ones already scheduled

    :param schedule: A dictionary of scheduled observations
    :param start_time:
    :param end_time:
    :return:
    """

    # If schedule is empty, there is no overlap
    if not schedule:
        return False

    # Check if this observation overlaps with the ones already scheduled
    for obs, time_range in schedule.iteritems():
        obs2_start = time_range[0] - OBS_PADDING
        obs2_end = time_range[1] + OBS_PADDING

        # Check for time ranges where duration is not defined:
        if not (obs2_end and end_time):
            # If there is an observation that has no end and starts before this observation
            if not obs2_end and obs2_start < start_time:
                return obs

            # If this observation starts before and does not end
            if not end_time and start_time < obs2_start:
                return obs

        # If this observation overlaps another time range
        elif (start_time <= obs2_end) and (obs2_start <= end_time):
            return obs

    # The observation start and end date does not overlap with any scheduled observation
    return False


def start_message(observation, start_time, end_time):
    """
    Build a human friendly message, indicating when the observation will start and for how long it will run

    :param observation: The observation parameters
    :param start_time: The start time of the observation
    :param end_time: The end time of the observation
    :return:
    """

    start_msg = "Observation {}, using the {} is scheduled to start NOW".format(
        observation['name'], observation['pipeline'])

    if 'start_time' in observation['config_parameters']:
        start_msg = "Observation {}, using the {} is scheduled to run at {:%Y-%m-%d %H:%M:%S}".format(
            observation['name'], observation['pipeline'], start_time)

    if end_time:
        start_msg += ' and will run for {} seconds'.format(observation['config_parameters']['duration'])
    else:
        start_msg += ' and will run indefinitely'


@click.command()
@click.option('--schedule', '-s', 'schedule_file_path', type=click.Path(exists=True), required=True,
              help='The scheduler json file')
def scheduler(schedule_file_path):
    """
    Schedule a series of observations

    :param schedule_file_path: The file path to the schedule
    :return:
    """

    with open(schedule_file_path) as json_data:
        scheduled_observations = json.load(json_data)
        message('INFO', 'Loaded schedule from {}'.format(schedule_file_path))

    # todo -- need to add validation of the config file

    s = sched.scheduler(time.time, time.sleep)
    now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    scheduler_time_range = {}
    for obs_name, observation in scheduled_observations.iteritems():
        # If no time is specified, start the pipeline in DEFAULT_WAIT_SECONDS seconds
        wait_seconds = DEFAULT_WAIT_SECONDS
        start_time = datetime.datetime.utcnow()
        end_time = None
        observation['name'] = obs_name
        if 'duration' in observation['config_parameters']:
            end_time = dateutil.parser.parse(observation['config_parameters']['start_time']) + datetime.timedelta(
                seconds=observation['config_parameters']['duration'])

        if 'start_time' in observation['config_parameters']:
            start_time = dateutil.parser.parse(observation['config_parameters']['start_time'])
            # Check that start time is valid
            wait_seconds = (start_time - now).total_seconds()

        # Check that start time is valid
        if wait_seconds < 1:
            message('INFO', "Scheduled start time for {} must be in the future".format(obs_name))
            sys.exit()

        overlapped_obs = obs_overlaps_schedule(scheduler_time_range, start_time, end_time)
        if overlapped_obs:
            message('ERROR', "The observation '{}' overlaps the time range specified in observation '{}'".format(
                        obs_name, overlapped_obs))
            sys.exit()

        scheduler_time_range[obs_name] = (start_time, end_time)
        message('INFO', start_message(observation, start_time, end_time))

        # Schedule the observation
        s.enter(delay=wait_seconds, priority=0, action=start_observation, argument=(observation,))

    message('INFO', 'Scheduler initialised. {} observations queued.'.format(len(scheduled_observations)))

    monitor = threading.Thread(target=monitoring_thread, args=(s,))
    monitor.daemon = True
    monitor.start()

    try:
        # Run scheduler
        s.run()

        #
    except KeyboardInterrupt:
        message('INFO', 'Ctrl-C received. Terminating the scheduler process.')
        sys.exit()

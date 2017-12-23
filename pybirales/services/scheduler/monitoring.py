import datetime
import dateutil.parser
import humanize
import logging as log
import time


def monitor_worker(scheduler):
    """
    Start the monitoring thread

    :param scheduler: The sched instance
    :return:
    """

    while not scheduler.empty():
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        for observation in scheduler.queue:
            obs_settings = observation.argument[0]
            start_time = dateutil.parser.parse(obs_settings['config_parameters']['start_time'])
            time_remaining = humanize.naturaltime(now - start_time)

            log.info('The %s for the "%s" observation is scheduled to start in %s',
                     obs_settings['pipeline'],
                     obs_settings['name'],
                     time_remaining)

        # Do not show the output again for the next N seconds
        time.sleep(60)

import datetime
import logging as log
import time

import dateutil.parser
import humanize
import pytz


def monitor_worker(scheduler):
    """
    Start the monitoring thread

    :param scheduler: The sched instance
    :return:
    """

    while not scheduler.empty():
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        for event in scheduler.queue:
            observation = event.argument[0]
            h_time_remaining = humanize.naturaltime(now - observation.start_time_padded)
            h_duration = humanize.naturaldelta(observation.duration)
            log.info('The %s for the `%s` observation is scheduled to start in %s and will run for %s',
                     observation.pipeline_name,
                     observation.name,
                     h_time_remaining, h_duration)

        # Do not show the output again for the next N seconds
        time.sleep(60)

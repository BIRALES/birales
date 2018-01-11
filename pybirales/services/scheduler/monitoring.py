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
    time_counter = 0
    while not scheduler.empty():
        # Process every N iterations
        if time_counter % 60 == 0:
            now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
            for event in scheduler.queue:
                observation = event.argument[0]
                h_time_remaining = humanize.naturaldelta(now - observation.start_time_padded)
                h_duration = humanize.naturaldelta(observation.duration)
                log.info('The %s for the `%s` observation is scheduled to start in %s and will run for %s',
                         observation.pipeline_name,
                         observation.name,
                         h_time_remaining, h_duration)

        time_counter += 1
        time.sleep(5)

    log.info('Monitoring thread terminated')

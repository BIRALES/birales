import datetime
import json
import logging
import logging as log
import threading
import time
from threading import Event

import yappi as profiler
from matplotlib import pyplot as plt

from pybirales import settings
from pybirales.pipeline.base.definitions import NoDataReaderException
from pybirales.repository.message_broker import pub_sub, broker

PIPELINE_CTL_CHL = 'birales_pipeline_control'
BIRALES_STATUS_CHL = 'birales_system_status'


def pipeline_status_worker(kill_pill):
    pub_sub = broker.pubsub()
    pub_sub.subscribe(PIPELINE_CTL_CHL)
    log.debug('Listening on #%s for messages', PIPELINE_CTL_CHL)

    for item in pub_sub.listen():
        if item['type'] == 'message':
            if item['data'] == 'KILL':
                log.info('KILL received on #{}. Killing pipeline'.format(PIPELINE_CTL_CHL))

                pub_sub.unsubscribe(PIPELINE_CTL_CHL)
        elif item['type'] == 'unsubscribe':
            log.debug('Un-subscribed from #%s', item['channel'])
            kill_pill.set()
            break
        elif item['type'] == 'subscribe':
            log.debug('Subscribed to #%s', item['channel'])
        else:
            log.warning('Could not handle the following message: %s', item)

    log.info('Pipeline status monitoring thread terminated')


class PipelineManager(object):
    """ Class to manage the pipeline """

    def __init__(self):
        # Class constructor
        self._modules = []
        self._plotters = []
        self._module_names = []

        # Get own configuration
        self._config = None
        self._enable_plotting = False
        self._plot_update_rate = 2
        if "manager" in settings.__dict__:
            self._config = settings.manager
            if "enable_plotting" in self._config.settings():
                self._enable_plotting = self._config.enable_plotting
            if "plot_update_rate" in self._config.settings():
                self._plot_update_rate = self._config.plot_update_rate

        self._stop = Event()

        self.count = 0
        self.name = None

        self._pipeline_controller = threading.Thread(target=pipeline_status_worker, args=(self._stop,),
                                                     name='Pipeline CTL')

        self._pipeline_controller.start()

    def add_module(self, name, module):
        """ Add a new module instance to the pipeline
        :param name: Name of the module instance
        :param module: Module instance
        """
        self._module_names.append(name)
        self._modules.append(module)

    def add_plotter(self, name, class_name, config, input_blob):
        """ Add a new plotter instance to the pipeline
        :param name: Name of the plotter instance
        :param class_name:
        :param config:
        :param input_blob:
        :return:
        """

        # Add plotter if plotting is enabled
        if self._enable_plotting:
            # Create the plotter instance
            plot = class_name(config, input_blob, plt.figure())

            # Create plotter indexing
            plot.index = plot.create_index()

            # Initialise the plotter and add to list
            plot.initialise_plot()
            self._plotters.append(plot)

    def start_pipeline(self, duration, observation):
        """
        Start running the pipeline

        :param duration: duration of observation in s (0 means run forever)
        :return:
        """

        try:
            logging.info("PyBIRALES: Starting")

            if settings.manager.profile:
                profiler.start()

            # Start all modules
            for module in self._modules:
                log.debug('Starting module {}'.format(module.name))
                module.start()

            self.wait_pipeline(duration=duration)

        except NoDataReaderException as exception:

            observation.model.status = 'finished'
        except KeyboardInterrupt:
            log.warning('Keyboard interrupt detected. Stopping the pipeline')
            observation.model.status = 'stopped'
        except Exception as exception:
            logging.exception('Pipeline error: %s', exception.__class__.__name__)
            observation.model.status = 'error'
        else:
            logging.info('Pipeline stopped without error')
            observation.model.status = 'finished'
        finally:
            observation.model.date_time_end = datetime.datetime.utcnow()
            observation.save()

            self.stop_pipeline()

    def stop_pipeline(self):
        """ Stop pipeline (one at a time) """
        self._stop.set()
        # Loop over all modules
        for module in self._modules:
            # Stop module
            module.stop()

            for t in threading.enumerate():
                if t.isAlive():
                    log.warning('Thread %s is alive', t.getName())

            # Try to kill it several time, otherwise skip (will be killed when main process exists)
            tries = 0
            while not module.is_stopped and tries < 5:
                time.sleep(0.5)
                tries += 1
                # All done

        log.info('Pipeline Manager stopped')

        if settings.manager.profile:
            profiler.stop()
            stats = profiler.get_func_stats()
            profiling_file_path = settings.manager.profiler_file_path + '_{:%Y%m%d_%H:%M}.stats'.format(
                datetime.datetime.utcnow())
            log.info('Profiling stopped. Dumping profiling statistics to %s', profiling_file_path)
            stats.save(profiling_file_path, type='callgrind')

        # kill listener thread
        log.debug('trying to kill pipeline %s', PIPELINE_CTL_CHL)
        broker.publish(PIPELINE_CTL_CHL, 'KILL')

    def is_module_stopped(self):
        for module in self._modules:
            if module.is_stopped:
                log.debug('Module {} has stopped'.format(module.name))
                return True
        return False

    @property
    def stopped(self):
        return self._stop.is_set()

    def all_modules_stopped(self):
        return all([module.is_stopped for module in self._modules])

    def wait_pipeline(self, duration=None):
        """
        Wait for modules to finish processing. If a module is stopped, the pipeline is
        stopped

        :param duration: duration of observation in s (Run forever if no time is specified)

        :return:
        """

        start_time = datetime.datetime.utcnow()

        # Specify the update interval
        dt = datetime.timedelta(seconds=5)

        # force an initial update by subtracting two deltas away
        last_updated = start_time - 2 * dt

        while not self.stopped:
            now = datetime.datetime.utcnow()

            # If one module stops without setting the stop bit (such as through a signal
            # handler, stop all the pipeline
            if self.is_module_stopped():
                # self.stop_pipeline()
                break

            # If all modules have stopped, we are ready
            elif self.all_modules_stopped():
                log.debug('All modules are stopped')
                break

            # If the observation duration has elapsed stop pipeline
            elif duration and (now - start_time).seconds > duration:
                logging.info("Observation run for the entire duration ({}s), stopping pipeline".format(duration))
                # self.stop_pipeline()
                break

            else:

                # Indicate to the user that the pipeline is running (every 5 counts)
                if now - last_updated > dt:
                    broker.publish(BIRALES_STATUS_CHL, json.dumps({
                        'pipeline': {
                            'status': 'running',
                            'timestamp': now.isoformat('T'),
                            'next_update': (now + dt).isoformat('T'),
                            'dt': dt.seconds
                        }
                    }))

                    last_updated = now

                # Suspend the loop for a short time
                time.sleep(1)

        # self.stop_pipeline()

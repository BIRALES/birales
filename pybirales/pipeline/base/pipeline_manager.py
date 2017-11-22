import logging
import signal
import time
import yappi as profiler
import logging as log

from matplotlib import pyplot as plt
from datetime import datetime
from pybirales import settings
from pybirales.pipeline.base.definitions import NoDataReaderException
from threading import Event
import os


class PipelineManager(object):
    """ Class to manage the pipeline """

    def __init__(self, config_file, debug=False):
        logging.info("PyBIRALES: Initialising")

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

        # Set interrupt signal handler
        signal.signal(signal.SIGINT, self._signal_handler)

        self.count = 0

    def _signal_handler(self, signum, frame):
        """
        Capturing interrupt signal

        :param signum:
        :param frame:
        :return:
        """
        if not self._stop.is_set():
            logging.info("Ctrl-C detected by process %s, stopping pipeline", os.getpid())

            self.stop_pipeline()

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

    def start_pipeline(self):
        """
        Start running the pipeline
        :return:
        """

        try:
            logging.info("PyBIRALES: Starting")

            if settings.manager.profile:
                profiler.start()

            # Start all modules
            for module in self._modules:
                module.start()

            # If we have any plotter, go to plotter loop, otherwise wait
            if len(self._plotters) > 0:
                self.plotting_loop()
            else:
                self.wait_pipeline()

        except NoDataReaderException as exception:
            logging.info('Data finished %s', exception.__class__.__name__)
            self.stop_pipeline()
        except Exception as exception:
            logging.exception('Pipeline error: %s', exception.__class__.__name__)
            # An error occurred, force stop all modules
            self.stop_pipeline()
        else:
            logging.info('Pipeline stopped')

    def plotting_loop(self):
        """ Plotting loop """
        while True:
            for plotter in self._plotters:
                plotter.update_plot()
                logging.info("{} updated".format(plotter.__class__.__name__))
            time.sleep(self._plot_update_rate)

    def stop_pipeline(self):
        """ Stop pipeline (one at a time) """
        self._stop.set()
        # Loop over all modules
        for module in self._modules:
            # Stop module
            module.stop()

            # Try to kill it several time, otherwise skip (will be killed when main process exists)
            tries = 0
            while not module.is_stopped and tries < 5:
                time.sleep(0.5)
                tries += 1
                # All done

        if settings.manager.profile:
            profiler.stop()
            stats = profiler.get_func_stats()
            profiling_file_path = settings.manager.profiler_file_path+'_{:%Y%m%d_%H:%M}.stats'.format(datetime.utcnow())
            log.info('Profiling stopped. Dumping profiling statistics to %s', profiling_file_path)
            stats.save(profiling_file_path, type='callgrind')

    def is_module_stopped(self):
        for module in self._modules:
            if module.is_stopped:
                return True
        return False

    def wait_pipeline(self):
        """
        Wait for modules to finish processing. If a module is stopped, the pipeline is
        stopped

        :return:
        """

        while True:
            if self.is_module_stopped():
                self.stop_pipeline()
                break
            else:
                # Suspend the loop for a short time
                time.sleep(1)

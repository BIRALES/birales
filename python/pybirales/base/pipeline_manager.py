import ast
import logging
import re
import signal
import time
import yappi as profiler
import configparser
import logging as log
from logging.config import fileConfig as set_log_config
from matplotlib import pyplot as plt
from datetime import datetime
from pybirales.base import settings
from pybirales.base.definitions import PipelineError, NoDataReaderException
from threading import Event


class PipelineManager(object):
    """ Class to manage the pipeline """

    def __init__(self, config_file, debug=False):
        logging.info("PyBIRALES: Initialising")

        # Class constructor
        self._modules = []
        self._plotters = []
        self._module_names = []

        # Config patters
        self._config_pattern = re.compile("^True|False|[0-9]+(\.[0-9]*)?$")

        # Load configuration file
        self._configure_pipeline(config_file)

        # Check configuration
        self._check_configuration()

        # Initialise logging
        self._initialise_logging(debug)

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

    # Capturing interrupt signal
    def _signal_handler(self, signum, frame):

        if not self._stop.is_set():
            logging.info("Ctrl-C detected, stopping pipeline")
            self.stop_pipeline()

    def _configure_pipeline(self, config_file):
        """ Parse configuration file and set pipeline
        :param config_file: Configuration file path
        """
        parser = configparser.ConfigParser()
        parser.read(unicode(config_file))

        # Temporary class to create section object in settings file
        class Section(object):
            def settings(self):
                return self.__dict__.keys()

        # Loop over all section in config file
        for key, value in parser._sections.items():
            # Create instance to inject into settings file
            instance = Section()

            # Loop over all config entries in section
            for k, v in value.items():

                # Check if value is a number of boolean
                if re.match(self._config_pattern, v) is not None:
                    setattr(instance, k, ast.literal_eval(v))

                # Check if value is a list
                elif re.match("^\[.*\]$", re.sub('\s+', '', v)):
                    setattr(instance, k, ast.literal_eval(v))

                # Otherwise it is a string
                else:
                    setattr(instance, k, v)

            # Add object instance to settings
            setattr(settings, key, instance)

    @staticmethod
    def _check_configuration():
        """ Check that an observation entry is in the config file and it contains the required information """
        if "observation" not in settings.__dict__:
            raise PipelineError("PipelineManager: observation section not found in configuration file")

        if {"start_center_frequency", "channel_bandwidth", "samples_per_second"} - set(settings.observation.settings()):
            raise PipelineError("PipelineManager: Missing keys in observation section "
                                "(need start_center_frequency, bandwidth")

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

    @staticmethod
    def _initialise_logging(debug):
        """ Initialise logging functionality """
        set_log_config(settings.manager.loggging_config_file_path)
        logger = logging.getLogger()
        # Logging level should be INFO by default
        logger.setLevel(logging.INFO)
        if debug:
            # Change Logging level to DEBUG
            logger.setLevel(logging.DEBUG)

    def wait_pipeline(self):
        """ Wait for modules to finish processing """

        for module in self._modules:
            while module.isAlive() and module.is_stopped is False:
                module.join(5)
            if module.isAlive():
                logging.warning("PipelineManager: Killing thread %s abruptly", module.name)


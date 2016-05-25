import ConfigParser
import ast
import re

import signal

from pybirales.base import settings
from sys import stdout
import logging
import time

from pybirales.base.definitions import PipelineError
from matplotlib import pyplot as plt


class PipelineManager(object):
    """ Class to manage the pipeline """

    def __init__(self, config_file):
        """ Class constructor """
        self._modules = []
        self._plotters = []
        self._module_names = []

        # Initialise logging
        self._initialise_logging()

        # Config patters
        self._config_pattern = re.compile("^True|False|[0-9]+(\.[0-9]*)?$")

        # Load configuration file
        self._configure_pipeline(config_file)

        # Check configuration
        self._check_configuration()

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

        # Capturing interrupt signal
        def signal_handler(sig, frame):
            logging.info("Ctrl-C detected, stopping pipeline")
            self.stop_pipeline()

        # Set interrupt signal handler
        signal.signal(signal.SIGINT, signal_handler)

    def _configure_pipeline(self, config_file):
        """ Parse configuration file and set pipeline
        :param config_file: Configuration file path
        """
        parser = ConfigParser.SafeConfigParser()
        parser.read(config_file)

        # Temporary class to create section object in settings file
        class Section(object):
            def settings(self):
                return self.__dict__.keys()

        # Loop over all section in config file
        for key, value in parser._sections.iteritems():
            # Create instance to inject into settings file
            instance = Section()

            # Loop over all config entries in section
            for k, v in value.iteritems():

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

    def _check_configuration(self):
        """ Check that an observation entry is in the config file and it contains the required information """
        if "observation" not in settings.__dict__:
            raise PipelineError("PipelineManager: observation section not found in configuration file")

        if {"start_center_frequency", "bandwidth"} - set(settings.observation.settings()):
            raise PipelineError("PipelineManager: Missing keys in observation section "
                                "(need start_center_frequency, bandwidth")

    def add_module(self, name, module):
        """ Add a new module instance to the pipeline
        :param name: Name of the module instance
        :param module: Module instance
        """
        self._module_names.append(name)
        self._modules.append(module)

    def add_plotter(self, name, classname, config, input_blob):
        """ Add a new plotter instance to the pipeline
        :param name: Name of the plotter instance
        :param plotter: Plotter instance
        :return:
        """
        # Add plotter if plotting is enabled
        if self._enable_plotting:

            # Create the plotter instance
            plot = classname(config, input_blob, plt.figure() )

            # Create plotter indexing
            plot.index = plot.create_index()

            # Initialise the plotter and add to list
            plot.initialise_plot()
            self._plotters.append(plot)

    def start_pipeline(self):
        """ Start running pipeline """
        try:
            # Start all modules
            for module in self._modules:
                module.start()

            # If we have any plotter, go to plotter loop, otherwise wait
            if len(self._plotters) > 0:
                self.plotting_loop()
            else:
                self.wait_pipeline()

        except Exception:
            # An error occurred, force stop all modules
            self.stop_pipeline()

    def plotting_loop(self):
        """ Plotting loop """
        while True:
            for plotter in self._plotters:
                plotter.update_plot()
            time.sleep(self._plot_update_rate)

    def stop_pipeline(self):
        """ Stop pipeline (one at a time) """

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

    @staticmethod
    def _initialise_logging():
        """ Initialise logging functionality """
        log = logging.getLogger('')
        log.setLevel(logging.INFO)
        str_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch = logging.StreamHandler(stdout)
        ch.setFormatter(str_format)
        log.addHandler(ch)

    def wait_pipeline(self):
        """ Wait for modules to finish processing """
        for module in self._modules:
            while module.isAlive() and module._stop is False:
                module.join(2)
            if module.isAlive():
                logging.warning("PipelineManager: Killing thread %s abruptly" % module.name)


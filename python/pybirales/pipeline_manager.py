import ConfigParser
import ast
import re

import signal

from pybirales.base import settings
from sys import stdout
import logging
import time

from pybirales.base.definitions import PipelineError


class PipelineManager(object):
    """ Class to manage the pipeline """

    def __init__(self, config_file):
        """ Class constructor """
        self._modules = []
        self._module_names = []

        # Initialise logging
        self._initialise_logging()

        # Config patters
        self._config_pattern = re.compile("^True|False|[0-9]+(\.[0-9]*)?$")

        # Load configuration file
        self._configure_pipeline(config_file)

        # Check configuration
        self._check_configuration()

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
            raise PipelineError("PipelineManager: observation section not foudn in configuration file")

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

    def add_plotter(self, name, plotter):
        """ Add a new plotter instance to the pipeline
        :param name: Name of the plotter instance
        :param plotter: Plotter instance
        :return:
        """
        self.add_module(name, plotter)

    def start_pipeline(self):
        """ Start running pipeline """
        try:
            # Start all modules
            for module in self._modules:
                module.start()
        except Exception:
            # An error occurred, force stop all modules
            self.stop_pipeline()

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


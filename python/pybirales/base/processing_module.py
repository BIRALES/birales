import logging
import threading
from abc import abstractmethod
from threading import Thread

import time


class ProcessingModule(Thread):
    """ Processing module """

    def __init__(self, config, input_blob=None):
        """ Class constructor
        :param config: Configuration object
        :param input_blob: Input data blob
        """
        # Call superclass constructor
        super(ProcessingModule, self).__init__()

        # Set module configuration
        self._config = config

        # Check if module will output a data blob
        self._no_output = False
        if "no_output" in config.settings():
            self._no_output = config.no_output

        # Set module input and output blobs
        self._input = input_blob

        if self._no_output:
            self._output = None
        else:
            self._output = self.generate_output_blob()

        # Stopping clause
        self._stop = False
        self._is_stopped = True

    @abstractmethod
    def generate_output_blob(self):
        """ Create the output blob to be used by the next module in the pipeline
        :return: Create data blob """
        pass

    @abstractmethod
    def process(self, obs_info, input_data, output_data):
        """ Abstract function which must be implemented by all subclasses which will be called in the
            thread's run loop
        :param obs_info: Observation information
        :param input_data: Input data block
        :param output_data: Output data block
        """
        pass

    def run(self):
        """ Thread body """
        self._is_stopped = False
        while not self._stop:
            # Get pointer to input data if required
            input_data, obs_info = None, None
            if self._input is not None:
                # This can be released immediately since, data has already been deep copied
                input_data, obs_info = self._input.request_read()

            # Get pointer to output data if required
            output_data = None
            if self._output is not None:
                output_data = self._output.request_write()

            # Perform required processing
            res = self.process(obs_info, input_data, output_data)
            if res is not None:
                obs_info = res

            # Release writer lock and update observation info if required
            if self._output is not None:
                self._output.release_write(obs_info)

            # Release reader lock
            if self._input is not None:
                self._input.release_read()

            # A short sleep tp force a context switch (since locks do not force one)
            time.sleep(0.001)

        self._is_stopped = True

    def stop(self):
        """ Stops the current thread """
        self._stop = True

    @property
    def is_stopped(self):
        """ Specified whether the thread body is running or not """
        return self._is_stopped

    @property
    def output_blob(self):
        """ Return output blob generate by module """
        return self._output

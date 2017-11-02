from abc import abstractmethod
from threading import Thread

import time


class Plotter(object):

    def __init__(self, config, input_blob, figure):
        """ Class constructor
        :param config: Configuration object
        :param input_blob: Input data blob
        """

        # Set module configuration
        self._config = config

        # Set module input and output blobs
        self._input = input_blob

        # Get plot update period
        self._update_period = 2
        if config is not None and 'update_period' in config.settings():
            self._update_period = config.update_period

        # Default index
        self._index = None
        self._figure = figure

        # Stopping clause
        self.daemon = True
        self._stop = False
        self._is_stopped = True

    def update_plot(self):
        """ Thread body """
        # Get snapshot from blob
        data, obs_info = self._input.get_snapshot(self._index)
        self.refresh_plot(data, obs_info)

    @abstractmethod
    def initialise_plot(self):
        pass

    @abstractmethod
    def create_index(self):
        pass

    @abstractmethod
    def refresh_plot(self, input_data, obs_info):
        pass

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, x):
        self._index = x

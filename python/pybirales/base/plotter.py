from abc import abstractmethod
from threading import Thread

import time


class Plotter(Thread):

    def __init__(self, config, input_blob):
        """ Class constructor
        :param config: Configuration object
        :param input_blob: Input data blob
        """

        # Call superclass
        super(Plotter, self).__init__()

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

        # Stopping clause
        self.daemon = True
        self._stop = False
        self._is_stopped = True

    def run(self):
        """ Thread body """
        # Initialise plot
        self.initialise_plot()
        self._index = self.create_index()

        # Loop until thread is stopped
        self._is_stopped = False
        while not self._stop:
            # Get snapshot from blob
            data, obs_info = self._input.request_snapshot(self._index)
            self.update_plot(data, obs_info)
            time.sleep(self._update_period)
        self._is_stopped = True

    @abstractmethod
    def initialise_plot(self):
        pass

    @abstractmethod
    def create_index(self):
        pass

    @abstractmethod
    def update_plot(self, input_data, obs_info):
        pass

    @property
    def index(self):
        return self._index

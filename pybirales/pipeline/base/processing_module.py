import logging

from abc import abstractmethod
from threading import Thread, Event
from pybirales.pipeline.base.definitions import NoDataReaderException, InputDataNotValidException
from pybirales import settings
import time
import logging as log
import threading


class Module(Thread):
    _valid_input_blobs = []

    def __init__(self, config=None, input_blob=None):
        """

        :param config:
        :param input_blob:
        """

        # Call superclass
        super(Module, self).__init__()

        # Set module configuration
        self._config = config

        # Check if module will output a data blob
        self._no_output = False
        if self._config is not None and "no_output" in config.settings():
            self._no_output = config.no_output

        # Set module input and output blobs
        self._input = input_blob

        if self._no_output:
            self._output = None
        else:
            self._output = self.generate_output_blob()

        # Stopping clause
        self.daemon = True
        self._stop = Event()

    @abstractmethod
    def generate_output_blob(self):
        """ Create the output blob to be used by the next module in the pipeline
        :return: Create data blob
        """
        pass

    def stop(self):
        """
        Stops the current thread if it is not stopped already
        :return:
        """

        if not self.is_stopped:
            logging.info('Stopping %s module', self.name)
            self._stop.set()

    def _validate_data_blob(self, input_blob, valid_blobs):
        if type(input_blob) not in valid_blobs:
            raise InputDataNotValidException("Input blob for {} should be ({}). Got a {} instead.".format(
                self.__class__.__name__,
                ', '.join([bt.__name__ for bt in valid_blobs]),
                input_blob.__class__.__name__))

    def _tear_down(self):
        """
        Gracefully terminate this module (to be overridden)
        :return:
        """
        pass

    @property
    def is_stopped(self):
        """ Specified whether the thread body is running or not """
        return self._stop.is_set()

    @property
    def output_blob(self):
        """ Return output blob generate by module """
        return self._output


class Generator(Module):
    @abstractmethod
    def generate_output_blob(self):
        """
        Create the output blob to be used by the next module in the pipeline's chain.
        This method is to be overridden by the corresponding sub classes
        :return:
        """

        pass

    @abstractmethod
    def start_generator(self):
        """ Starts the generator. This is called by run """
        pass

    def run(self):
        """
        Thread body
        :return:
        """

        # Clear stop
        self._stop.clear()

        # Start generator
        self.start_generator()

        # Wait for stop command
        while not self._stop.is_set():
            time.sleep(1)

        self._stop.set()

    def request_output_blob(self):
        """
        Request a new output blob to write data into
        :return:
        """

        return self._output.request_write()

    def release_output_blob(self, obs_info):
        """
        Release the output blob
        :param obs_info: Observation information object
        :return:
        """

        self._output.release_write(obs_info)


class ProcessingModule(Module):
    """ Processing module """

    def __init__(self, config, input_blob):
        super(ProcessingModule, self).__init__(config, input_blob)
        logging.info('Initialised the %s module', self.__class__.__name__)

    @abstractmethod
    def generate_output_blob(self):
        """
        Create the output blob to be used by the next module in the pipeline
        :return:
        """
        pass

    @abstractmethod
    def process(self, obs_info, input_data, output_data):
        """
        Abstract function which must be implemented by all subclasses
        which will be called in the thread's run loop
        :param obs_info: Observation information
        :param input_data: Input data block
        :param output_data: Output data block
        :return:
        """
        pass

    def clean(self):
        pass

    def run(self):
        """ Thread body """
        while not self._stop.is_set():

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
            try:
                s = time.time()
                res = self.process(obs_info, input_data, output_data)
                tt = time.time() - s
                if tt < settings.receiver.nsamp / settings.observation.samples_per_second:
                    log.info('%s finished in %0.3f s', self.name, tt)
                else:
                    log.warning('%s finished in %0.3f s', self.name, tt)

                if res is not None:
                    obs_info = res
            except NoDataReaderException:
                logging.info("Data finished")
                self.stop()
            except OSError:
                log.exception("An OS exception has occurred. Stopping the pipeline")
                self.stop()

            # Release writer lock and update observation info if required
            if self._output is not None:
                self._output.release_write(obs_info)

            # Release reader lock
            if self._input is not None:
                self._input.release_read()

            # A short sleep to force a context switch (since locks do not force one)
            time.sleep(0.001)

        # Clean
        self._tear_down()
        log.info('%s killed [Active threads: %s]', self.name, self._get_active_threads())

    def _get_active_threads(self):
        """
        Return the threads that are still active

        :return:
        """
        main_thread = threading.current_thread()
        active_threads = []
        for t in threading.enumerate():
            if t is main_thread:
                continue
            active_threads.append(t.getName())
        return ",".join(map(str, active_threads))

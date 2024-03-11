import logging
import logging as log
import threading
import time
from abc import abstractmethod
from threading import Thread, Event

from pybirales import settings
from pybirales.pipeline.base.definitions import NoDataReaderException, InputDataNotValidException, \
    BIRALESObservationException


class Module(Thread):
    _valid_input_blobs = []

    def __init__(self, config=None, input_blob=None):
        """

        :param config:
        :param input_blob:
        """

        # Call superclass
        super(Module, self).__init__()

        # Generate a unique identifier for the module instance
        self._unique_id = id(self)

        # Set module configuration
        self._config = config

        # Check if module will output a data blob
        self._no_output = False
        if self._config is not None and "no_output" in config.settings():
            self._no_output = config.no_output

        # Set module input and associate blob with module
        self._input = input_blob
        if self._input is not None:
            self._input.add_reader(self._unique_id)

        if self._no_output:
            self._output = None
        else:
            self._output = self.generate_output_blob()

        # Stopping clause
        self.daemon = True
        self._stop_module = Event()

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
            self._stop_module.set()

            # self._tear_down()

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
        return self._stop_module.is_set()

    @property
    def output_blob(self):
        """ Return output blob generate by module """
        return self._output

    @staticmethod
    def _get_active_threads():
        """
        Return the threads that are still active

        :return:
        """
        main_thread = threading.current_thread()
        active_threads = []
        for t in threading.enumerate():
            if t is main_thread:
                continue
            active_threads.append(t.name)
        return ",".join(map(str, active_threads))


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
        self._stop_module.clear()

        # Start generator
        self.start_generator()

        # Wait for stop command
        while not self._stop_module.is_set():
            time.sleep(1)

        self._stop_module.set()

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
        self._iter_count = 1
        self._is_offline = settings.manager.offline
        logging.info('Initialised the %s module', self.__class__.__name__)

    def set_name(self, module_name):
        """ Set the name of the module to the thread"""
        self.name = module_name

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
        while not self._stop_module.is_set():

            # Get pointer to input data if required
            input_data, obs_info = None, {'stop_pipeline_at': -1}

            if self._input is not None:
                # This can be released immediately since data has already been deep copied
                while input_data is None and not self._stop_module.is_set():
                    input_data, obs_info = self._input.request_read(self._unique_id, timeout=0.1)

                if self._stop_module.is_set():
                    break

                if 'stop_pipeline_at' not in obs_info:
                    obs_info['stop_pipeline_at'] = -1

            # Get pointer to output data if required
            output_data = None
            if self._output is not None:
                while output_data is None and not self._stop_module.is_set():
                    output_data = self._output.request_write(timeout=10)

                if self._stop_module.is_set():
                    break

            # Perform required processing
            try:
                s = time.time()

                if obs_info['stop_pipeline_at'] == self._iter_count:
                    log.info('Stop pipeline message broadcast to the %s module', self.name)
                    self.stop()
                else:
                    res = self.process(obs_info, input_data, output_data)
                    if res is not None:
                        obs_info = res
                tt = time.time() - s

                log.info('[Iteration {}] {} finished in {:0.3f}s'.format(self._iter_count, self.name, tt))

            except BIRALESObservationException:
                log.exception("A Birales exception has occurred. Stopping the pipeline")
                self.stop()
            except NoDataReaderException:
                log.info("Data Finished")
                self.stop()
            except OSError:
                log.exception("An OS exception has occurred. Stopping the pipeline")
                self.stop()

            # Release writer
            if self._output is not None:
                self._output.release_write(obs_info)

            # Release reader
            if self._input is not None:
                self._input.release_read(self._unique_id)

            # A short sleep to force a context switch (since locks do not force one)
            time.sleep(0.001)

            self._iter_count += 1

        # Clean
        self._tear_down()

        self._stop_module.set()

        log.info('%s killed [Active threads: %s]', self.name, self._get_active_threads())

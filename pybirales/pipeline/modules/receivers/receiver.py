import ctypes
import datetime
import json
import logging
import os
import sys
import numpy as np
from enum import Enum

from pybirales import settings
from pybirales.pipeline.base.definitions import PipelineError, ObservationInfo
from pybirales.pipeline.base.processing_module import Generator
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob
from pybirales.repository.message_broker import broker
from pybirales.services.instrument.backend import Backend

np.set_printoptions(threshold=sys.maxsize)


class Complex64t(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float)]


class Result(Enum):
    """ Result enumeration """
    Success = 0
    Failure = -1
    ReceiverUninitialised = -2
    ConsumerAlreadyInitialised = -3
    ConsumerNotInitialised = -4


class LogLevel(Enum):
    """ Log level """
    Fatal = 1
    Error = 2
    Warning = 3
    Info = 4
    Debug = 5


def logging_callback(level, message):
    """ Wrapper to logging function in DAQ """
    if level == LogLevel.Fatal.value:
        logging.fatal(message)
    elif level == LogLevel.Error.value:
        logging.error(message)
    elif level == LogLevel.Warning.value:
        logging.warning(message)
    elif level == LogLevel.Info.value:
        logging.info(message)
    elif level == LogLevel.Debug.value:
        logging.debug(message)


class Receiver(Generator):
    """ Receiver """

    def __init__(self, config, input_blob=None):
        # This module does not need an input block
        self._validate_data_blob(input_blob, valid_blobs=[type(None)])

        # Sanity checks on configuration
        if {'nsamp', 'nants', 'nsubs', 'port', 'interface', 'frame_size',
            'frames_per_block', 'nblocks', 'nbits', 'complex', 'npols'} - set(config.settings()) != set():
            raise PipelineError("Receiver: Missing keys on configuration "
                                "(nsamp, nants, nsubs, npols, ports, interface, frame_size, frames_per_block, nblocks)")
        self._nsamp = config.nsamp
        self._nants = config.nants
        self._nsubs = config.nsubs
        self._nbits = config.nbits
        self._npols = config.npols
        self._complex = config.complex
        self._samples_per_second = settings.observation.samples_per_second

        self._read_count = 0
        self._metrics_poll_freq = 10
        self._metric_channel = 'antenna_metrics'

        # Define data type
        if self._nbits == 64 and self._complex:
            self._datatype = np.complex64
        else:
            raise PipelineError("Receiver: Unsupported data type (bits, complex)")

        # Call superclass initialiser
        super(Receiver, self).__init__(config, input_blob)

        # Set global pointer to receiver to be used by the DAQ callback
        global receiver_instance
        receiver_instance = self

        # Initialise DAQ
        self._callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_void_p), ctypes.c_double, ctypes.c_uint32,
                                               ctypes.c_uint32)
        self._logger_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)
        self._daq = None
        self._daq_success = 0
        self._callback = self._get_callback_function()

        self._birales_library = None

        # Processing module name
        self.name = "Receiver"

    def generate_output_blob(self):
        """ Generate output data blob """
        # Generate blob
        return ReceiverBlob(self._config, [('npols', self._npols),
                                           ('nsubs', self._nsubs),
                                           ('nsamp', self._nsamp),
                                           ('nants', self._nants)],
                            datatype=self._datatype)

    def start_generator(self):
        """ Start receiving data """
        backend = Backend.Instance()
        self._initialise_library()
        self._initialise_receiver(backend.read_startup_time())

    def stop(self):
        """ Stop generator """
        logging.info('Stopping the receiver module')

        if self._daq:
            # print(self._daq.stopConsumer("birales"), Result.Failure.value, self._daq.stopReceiver())
            consumer_stopped = self._daq.stopConsumer("birales")
            receiver_stopped = self._daq.stopReceiver()

            logging.debug(
                'DAQ Birales consumer stopped: {}. DAQ Receiver stopped: {}'.format(consumer_stopped, receiver_stopped))

            if consumer_stopped != Result.Failure.value and receiver_stopped != Result.Failure.value:
                roach = Backend.Instance().roach

                logging.debug('ROACH is connected: {}. Disconnecting.'.format(roach.is_connected()))

                if roach.is_connected():
                    roach.stop()

                logging.debug('ROACH is connected: {}'.format(roach.is_connected()))

                # Backend.Instance()._roach.disconnect()
                self._stop.set()
            else:
                logging.critical("Failed to stop Receiver!")

    def _get_callback_function(self):
        def data_callback(data, timestamp, arg1, arg2):
            """ Data callback
            :param data: Data pointer
            :param timestamp: timestamp of first sample in the data """

            # Calculate number of value to process
            nof_values = settings.receiver.nsubs * settings.receiver.nants * settings.receiver.nsamp
            buffer_from_memory = ctypes.pythonapi.PyBuffer_FromMemory
            buffer_from_memory.restype = ctypes.py_object
            values = buffer_from_memory(data, np.dtype(np.complex64).itemsize * nof_values)
            values = np.frombuffer(values, np.complex64)

            obs_info = ObservationInfo()
            obs_info['sampling_time'] = 1.0 / settings.observation.samples_per_second
            obs_info['start_center_frequency'] = settings.observation.start_center_frequency
            obs_info['channel_bandwidth'] = settings.observation.channel_bandwidth

            try:
                obs_info['timestamp'] = datetime.datetime.utcfromtimestamp(timestamp)
            except ValueError:
                logging.warning('A Timestamp error occurred in the receiver')
                obs_info['timestamp'] = datetime.datetime.utcnow()
                logging.warning('An error has occurred when reading timestamp %s. Will use %s.',
                                datetime.datetime.utcfromtimestamp(timestamp), obs_info['timestamp'])
                pass
            obs_info['nsubs'] = self._nsubs
            obs_info['nsamp'] = self._nsamp
            obs_info['nants'] = self._nants
            obs_info['npols'] = self._npols

            obs_info['transmitter_frequency'] = settings.observation.transmitter_frequency

            # Get output blob
            output_data = self.request_output_blob()

            # Copy data to output buffer
            output_data[:] = values.reshape((obs_info['nsubs'], obs_info['nsamp'], obs_info['nants']))

            # Release output blob
            self.release_output_blob(obs_info)

            # Calculate RMS
            # self._calculate_rms(output_data)

            logging.info("Receiver: Received buffer ({})".format(obs_info['timestamp'].time()))

            # Publish the RMS voltages
            self.publish_antenna_metrics(self._read_count, output_data, obs_info)

            self._read_count += 1

        return self._callback_type(data_callback)

    @staticmethod
    def _calculate_rms(input_data):
        """ Calculate the RMS of the incoming antenna data
        :param input_data: Input antenna data """
        rms_values = np.mean(np.sum(np.power(input_data, 2), axis=1), 1)[0]

        return np.squeeze(np.sqrt(np.sum(np.power(np.abs(input_data), 2.), axis=2)))

    def _initialise_receiver(self, start_time):
        """ Initialise the receiver """

        # Set logging callback - disabled (causing seg fault)
        # self._daq.attachLogger(self._logger_callback(self._logging_callback))

        # Configure receiver
        if self._daq.startReceiver(self._config.interface,
                                   self._config.ip,
                                   self._config.frame_size,
                                   self._config.frames_per_block,
                                   self._config.nblocks) != Result.Success.value:
            raise PipelineError("Receiver: Failed to start")

        # Set receiver ports
        if self._daq.addReceiverPort(self._config.port) != Result.Success.value:
            raise PipelineError("Receiver: Failed to set receiver port %d" % self._config.port)

        # Generate configuration for raw consumer
        params = {"nof_antennas": self._nants,
                  "nof_samples": self._nsamp,
                  "start_time": start_time,
                  "samples_per_second": self._samples_per_second}

        # Load birales data consumer
        if self._daq.loadConsumer(self._birales_library, "birales") != self._daq_success:
            raise PipelineError("Failed to load birales consumer")

        # Initialise birales consumer
        if self._daq.initialiseConsumer("birales", json.dumps(params)) != self._daq_success:
            raise PipelineError("Failed to initialise birales consumer")

        # Start birales consumer
        if self._daq.startConsumer("birales", self._callback) != self._daq_success:
            raise PipelineError("Failed to start birales consumer")

    def _initialise_library(self):
        """ Initialise DAQ library """

        # Load library
        self._daq = ctypes.CDLL(settings.receiver.daq_file_path)

        # Define attachLogger
        self._daq.attachLogger.argtypes = [self._logger_callback]
        self._daq.attachLogger.restype = None

        # Define startReceiver function
        self._daq.startReceiver.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32,
                                            ctypes.c_uint32]
        self._daq.startReceiver.restype = ctypes.c_int

        # Define stopReceiver function
        self._daq.stopReceiver.argtypes = []
        self._daq.stopReceiver.restype = ctypes.c_int

        # Define loadConsumer function
        self._daq.loadConsumer.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._daq.loadConsumer.restype = ctypes.c_int

        # Define initialiseConsumer function
        self._daq.initialiseConsumer.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._daq.initialiseConsumer.restype = ctypes.c_int

        # Define startConsumer function
        self._daq.startConsumer.argtypes = [ctypes.c_char_p, self._callback_type]
        self._daq.startConsumer.restype = ctypes.c_int

        # Define stopConsumer function
        self._daq.stopConsumer.argtypes = [ctypes.c_char_p]
        self._daq.stopConsumer.restype = ctypes.c_int

        # Locate libbirales.so
        self._birales_library = self._find_in_path("libbirales.so", "/usr/local/lib")

    @staticmethod
    def _logging_callback(level, message):
        """ Wrapper to logging function in DAQ """
        if level == LogLevel.Fatal.value:
            logging.fatal(message)
        elif level == LogLevel.Error.value:
            logging.error(message)
        elif level == LogLevel.Warning.value:
            logging.warning(message)
        elif level == LogLevel.Info.value:
            logging.info(message)
        elif level == LogLevel.Debug.value:
            logging.debug(message)

    @staticmethod
    def _find_in_path(name, path):
        """ Find a file in a path
        :param name: File name
        :param path: Path to search in """
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)

        return None

    def publish_antenna_metrics(self, iteration, data, obs_info):
        if iteration % self._metrics_poll_freq == 0:
            rms_voltages = self._calculate_rms(data).tolist()
            timestamp = obs_info['timestamp'].isoformat('T')

            msg = json.dumps({'timestamp': timestamp,
                              'voltages': rms_voltages})
            broker.publish(self._metric_channel, msg)

            logging.debug('Published antenna metrics %s: %s', timestamp,
                          ', '.join(['%0.2f'] * len(rms_voltages)) % tuple(rms_voltages))

            logging.debug('Delay between server and roach is ~ {:0.2f} seconds'.format(
                (datetime.datetime.utcnow() - obs_info['timestamp']).total_seconds()))


if __name__ == "__main__":
    from psutil import virtual_memory
    import time


    def run(r, i):
        print 'Start receiver', i, virtual_memory().total
        r.start_generator()
        time.sleep(2)

        print 'Receiver', i, ' started', virtual_memory().total
        time.sleep(2)

        r.stop()
        print 'Stop receiver', i, virtual_memory().total
        time.sleep(3)


    class config:
        class receiver:
            nsamp = 262144
            nants = 32
            nsubs = 1
            nbits = 64
            npols = 1
            complex = True


    r = Receiver(config=config.receiver)

    run(r, 1)

    run(r, 2)

    run(r, 3)

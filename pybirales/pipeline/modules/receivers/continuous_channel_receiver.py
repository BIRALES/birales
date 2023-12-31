import ctypes
import logging
import numpy as np
from enum import Enum
from pybirales import settings
from pybirales.pipeline.base.definitions import PipelineError, ObservationInfo
from pybirales.pipeline.base.processing_module import Generator
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob

np.set_printoptions(threshold=np.nan)


class Complex8t(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int8),
                ("y", ctypes.c_int8)]


# Custom numpy type for creating complex signed 8-bit data
complex_8t = np.dtype([('real', np.int8), ('imag', np.int8)])


class Result(Enum):
    """ Result enumeration """
    Success = 0
    Failure = -1
    ReceiverUninitialised = -2
    ConsumerAlreadyInitialised = -3
    ConsumerNotInitialised = -4


class ContinuousChannelReceiver(Generator):
    """ Receiver """

    def __init__(self, config, input_blob=None):
        # This module does not need an input block
        self._validate_data_blob(input_blob, valid_blobs=[type(None)])

        # Sanity checks on configuration
        if {'nsamp', 'nants', 'nsubs', 'port', 'npols', 'interface', 'frame_size',
            'frames_per_block', 'nblocks'} - set(config.settings()) != set():
            raise PipelineError("Receiver: Missing keys on configuration "
                                "(nsamp, nants, nsubs, ports, npols, interface, frame_size, frames_per_block, nblocks)")
        self._nsamp = config.nsamp
        self._nants = config.nants
        self._nsubs = config.nsubs
        self._npols = config.npols
        self._samples_per_second = settings.observation.samples_per_second
        self._start_time = 0

        # Define data type
        self._datatype = np.complex64

        # Call superclass initialiser
        super(ContinuousChannelReceiver, self).__init__(config, input_blob)

        # Set global pointer to receiver to be used by the DAQ callback
        global receiver_instance
        receiver_instance = self

        # Initialise DAQ
        self._callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(Complex8t), ctypes.c_double)
        self._daq = None
        self._initialise_library()
        self._initialise_receiver()

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
        self._initialise_library()
        self._initialise_receiver()

    def _get_callback_function(self):
        def data_callback(data, timestamp):
            """ Data callback
            :param data: Data pointer
            :param timestamp: timestamp of first sample in the data """

            # Calculate number of value to process
            nof_values = settings.continuous_channel_receiver.npols * settings.continuous_channel_receiver.nsubs * \
                         settings.continuous_channel_receiver.nants * settings.continuous_channel_receiver.nsamp

            buffer_from_memory = ctypes.pythonapi.PyBuffer_FromMemory
            buffer_from_memory.restype = ctypes.py_object
            values = buffer_from_memory(data, complex_8t.itemsize * nof_values)
            values = np.frombuffer(values, complex_8t)
            # values = memoryview(data)

            obs_info = ObservationInfo()
            obs_info['sampling_time'] = 1.0 / settings.observation.samples_per_second
            obs_info['start_center_frequency'] = settings.observation.start_center_frequency
            obs_info['channel_bandwidth'] = settings.observation.channel_bandwidth
            obs_info['timestamp'] = timestamp
            obs_info['npols'] = self._npols
            obs_info['nsubs'] = self._nsubs
            obs_info['nsamp'] = self._nsamp
            obs_info['nants'] = self._nants

            # Get output blob
            output_data = self.request_output_blob()

            # TODO: tranpose data to fit our needs

            # Copy data to output buffer
            output_data[:] = values.reshape(
                (obs_info['npols'], obs_info['nsubs'], obs_info['nsamp'], obs_info['nants']))

            # Release output blob
            self.release_output_blob(obs_info)

            logging.info("Receiver: Received buffer")

        return self._callback_type(data_callback)

    def _initialise_receiver(self):
        """ Initialise the receiver """

        # Configure receiver
        self._daq.setReceiverConfiguration(self._nants, self._nsubs, self._nsubs, 1, 1, self._npols)

        # Start receiver
        if self._daq.startReceiver(self._config.interface.encode(),
                                   self._config.frame_size,
                                   self._config.frames_per_block,
                                   self._config.nblocks) != Result.Success.value:
            raise PipelineError("Receiver: Failed to start")

        # Set receiver ports
        if self._daq.addReceiverPort(self._config.port) != Result.Success.value:
            raise PipelineError("Receiver: Failed to set receiver port %d" % self._config.port)

        # Start consumer
        if self._daq.startContinuousChannelConsumer(self._nsamp, 16, 32) != Result.Success.value:
            raise Exception("Failed to start continuous channel data consumer")

        # Set channel data consumer callback
        self._callback = self._get_callback_function()
        if self._daq.setContinuousChannelConsumerCallback(self._callback) != Result.Success.value:
            raise PipelineError("Receiver: Failed to set consumer callback")

        logging.info("Waiting for incoming data")

    def _initialise_library(self):
        """ Initialise DAQ library """

        # Load library
        self._daq = ctypes.CDLL("/opt/aavs/lib/libaavsdaq.so")

        # Define setReceiverConfiguration function
        self._daq.setReceiverConfiguration.argtypes = [ctypes.c_uint16, ctypes.c_uint16,
                                                       ctypes.c_uint16, ctypes.c_uint8,
                                                       ctypes.c_uint8, ctypes.c_uint8]
        self._daq.setReceiverConfiguration.restype = None

        # Define startReceiver function
        self._daq.startReceiver.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._daq.startReceiver.restype = ctypes.c_int

        # Define addReceiverPort function
        self._daq.addReceiverPort.argtypes = [ctypes.c_uint16]
        self._daq.addReceiverPort.restype = ctypes.c_int

        # Define setContinuousChannelConsumerCallback function
        self._daq.setContinuousChannelConsumerCallback.argtypes = [self._callback_type]
        self._daq.setContinuousChannelConsumerCallback.restype = ctypes.c_int

        # Define startContinuousChannelConsumer function
        self._daq.startContinuousChannelConsumer.argtypes = [ctypes.c_uint32, ctypes.c_uint16, ctypes.c_uint16]
        self._daq.startContinuousChannelConsumer.restype = ctypes.c_int

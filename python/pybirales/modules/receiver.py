import ctypes
import logging
import numpy as np
from enum import Enum

from pybirales.base import settings
from pybirales.base.definitions import PipelineError, ObservationInfo
from pybirales.base.processing_module import Generator
from pybirales.blobs.receiver_data import ReceiverBlob


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


class Receiver(Generator):
    """ Dummy data generator """

    def __init__(self, config, input_blob=None):

        # This module does not need an input block
        if input_blob is not None:
            raise PipelineError("Receiver: Receiver does not need an input block")

        # Sanity checks on configuration
        if {'nsamp', 'nants', 'nsubs', 'port', 'interface', 'frame_size',
            'frames_per_block', 'nblocks', 'nbits', 'complex'}  - set(config.settings()) != set():
            raise PipelineError("Receiver: Missing keys on configuration "
                                "(nsamp, nants, nsubs, ports, interface, frame_size, frames_per_block, nblocks)")
        self._nsamp = config.nsamp
        self._nants = config.nants
        self._nsubs = config.nsubs
        self._nbits = config.nbits
        self._complex = config.complex

        # Define data type
        if self._nbits == 64 and self._complex:
            self._datatype = np.complex64
        else:
            raise PipelineError("DummyDataGenerator: Unsupported datatype (bits, complex)")

        # Call superclass initialiser
        super(Receiver, self).__init__(config, input_blob)

        # Initialise DAQ
        self._callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(Complex64t), ctypes.c_double)
        self._daq = None
        self._initialise_library()
        self._initialise_receiver()

        # Processing module name
        self.name = "Receiver"

    def generate_output_blob(self):
        """ Generate output data blob """
        # Generate blob
        return ReceiverBlob(self._config, [('nsubs', self._nsubs),
                                           ('nsamp', self._nsamp),
                                           ('nants', self._nants)],
                            datatype=self._datatype)

    def _initialise_receiver(self):
        """ Initialise the receiver """

        # Configure receiver
        self._daq.setReceiverConfiguration(self._nants, 1, 1, 1, 1, 1, 0)

        # Start receiver
        if self._daq.startReceiver(self._config.interface,
                                   self._config.frame_size,
                                   self._config.frames_per_block,
                                   self._config.nblocks) != Result.Success.value:
            raise PipelineError("Receiver: Failed to start")

        # Set receiver ports
        if self._daq.addReceiverPort(self._config.port) != Result.Success.value:
            raise PipelineError("Receiver: Failed to set receiver port %d" % self._config.port)

        # Start data consumer
        if self._daq.startBiralesConsumer(self._nsamp) != Result.Success.value:
            raise PipelineError("Receiver: Failed to start data consumer")

        # Set channel data consumer callback
        a = self._daq.setBiralesConsumerCallback(self._callback_type(self._data_callback))
        if a != Result.Success.value:
            raise PipelineError("Receiver: Failed to set consumer callback %s" % a)

    def _data_callback(self, data, timestamp):
        """ Data callback
        :param data: Data pointer
        :param timestamp: timestamp of first sample in the data """

        # Calculate number of value to process
        nof_values = settings.receiver.nants * settings.receiver.nsamp

        buffer_from_memory = ctypes.pythonapi.PyBuffer_FromMemory
        buffer_from_memory.restype = ctypes.py_object

        values = buffer_from_memory(data, np.dtype(np.complex64).itemsize * nof_values)
        values = np.frombuffer(values, np.complex64)

        obs_info = ObservationInfo()
        obs_info['sampling_time'] = 0.0
        obs_info['timestamp'] = 0.0
        obs_info['nsubs'] = self._nsubs
        obs_info['nsamp'] = self._nsamp
        obs_info['nants'] = self._nsubs

        # Get output blob
        output_data = self.request_output_blob()

        # Release output blob
        self.release_output_blob(obs_info)

        logging.info("Receiver: Received and cast data")

    def _initialise_library(self):
        """ Initialise DAQ library """

        # Load library
        self._daq = ctypes.CDLL("libaavsdaq.so")

        # Define setReceiverConfiguration function
        self._daq.setReceiverConfiguration.argtypes = [ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint8, ctypes.c_uint8,
                                                       ctypes.c_uint16, ctypes.c_uint8, ctypes.c_uint16]
        self._daq.setReceiverConfiguration.restype = None

        # Define startReceiver function
        self._daq.startReceiver.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._daq.startReceiver.restype = ctypes.c_int

        # Define addReceiverPort function
        self._daq.addReceiverPort.argtypes = [ctypes.c_uint16]
        self._daq.addReceiverPort.restype = ctypes.c_int

        # Define startBeamConsumer function
        self._daq.startBiralesConsumer.argtypes = [ctypes.c_uint32]
        self._daq.startBiralesConsumer.restype = ctypes.c_int

        # Define setBeamConsumerCallback function
        self._daq.setBiralesConsumerCallback.argtypes = [self._callback_type]
        self._daq.setBiralesConsumerCallback.restype = ctypes.c_int

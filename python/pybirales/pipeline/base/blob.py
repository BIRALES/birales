import copy
import logging
import numpy as np
import threading
import time
from multiprocessing import Lock

from pybirales.pipeline.base.definitions import ObservationInfo


class DataBlob:
    """ Data blob super class """

    def __init__(self, shape, datatype, nof_blocks=3):
        """ Class constructor
        :param shape: The shape of the underlying datatype
        :param datatype: Datatype of blob data
        :param nof_blocks: Number of data blocks in DataBlob
        """

        # Shape and data type of the data
        self._shape = shape
        self._data_type = datatype

        # The data blob as a numpy array
        self._data = self.initialise(nof_blocks, datatype)

        # Create observation info object for every data block
        self._obs_info = []

        # Counter to determine whether a block in the blob is written or not. When a blob is written to,
        # the value of the counter set to the number of readers. When each reader processes the blob, it
        # decrements it
        self._remaining_readers = [0 for _ in range(nof_blocks)]

        # Generate a default observation information object per block
        self._obs_info = [ObservationInfo() for _ in range(nof_blocks)]

        # The number of blocks in the data blob
        self._nof_blocks = nof_blocks

        # Empty dictionary that will contain reader information
        self._nof_readers = 0
        self._readers = {}

        # Current writer index
        self._writer_index = 0

        # Lock for data structures shared between readers and writers
        self._shared_lock = Lock()

    def initialise(self, nof_blocks, datatype):
        """ Initialise the data blob. Allows subclasses to override this to change how and where
        the buffer are initialised
        :param nof_blocks: Number of data blocks in DataBlob
        :param datatype: Datatype of blob data """

        # The data blob as a numpy array
        data_shape = []
        for item in self._shape:
            data_shape.append(int(item[1]))
        return np.zeros((nof_blocks,) + tuple(data_shape), dtype=datatype)

    def add_reader(self, identifier):
        """ Add a new reader that will use the data blob.
        :param identifier: The unique identifier of the reader to associate the correct reader index """
        self._readers[identifier] = {'index': 0, 'reading': False}
        self._nof_readers += 1

    def request_read(self, identifier, timeout=None):
        """ Wait while reader and writer are not pointing to the same block, then return read block
        :return: Data splice associated with current block and associated observation information
        """

        # Check if a valid identifier was specified
        if identifier not in self._readers.keys():
            logging.error(f"Invalid reader identifier specified for {self.__class__.__name__}")
            exit()

        # Extract reader index
        reader_index = self._readers[identifier]['index']

        # Start timing
        t_start = time.time()

        # The reader must wait for the writer to insert data into the blob. When the reader and writer have the
        # same index it means that the reader is waiting for data to be available (the writer must be ahead of reader)
        self._shared_lock.acquire()
        while reader_index == self._writer_index and self._remaining_readers[reader_index] == 0:
            self._shared_lock.release()
            time.sleep(0.05)  # Wait for data to become available

            # If timeout has elapsed, return None
            if timeout and time.time() - t_start >= timeout:
                return None, None

            # Acquire shared lock before re-attempting check
            self._shared_lock.acquire()

        # Update reader information to specify that current reader is reading
        self._readers[identifier]['reading'] = True

        # Release shared lock
        self._shared_lock.release()

        # Return data splice
        return self._data[reader_index], copy.copy(self._obs_info[reader_index])

    def release_read(self, identifier):
        """ Finished reading data block, increment reader index and release lock """

        # Check if a valid identifier was specified
        if identifier not in self._readers.keys():
            logging.error(f"Invalid reader identifier specified for {self.__class__.__name__}")
            exit()

        # Extract reader index
        reader_index = self._readers[identifier]['index']

        # Acquire shared lock and update reader metadata
        self._shared_lock.acquire()
        self._remaining_readers[reader_index] -= 1
        self._readers[identifier]['reading'] = False
        self._readers[identifier]['index'] = (reader_index + 1) % self._nof_blocks
        self._shared_lock.release()

    def request_write(self, timeout=None):
        """ Get next block for writing
        :return: Data splice associated with current block
        """
        t_start = time.time()

        # Acquire shared lock
        self._shared_lock.acquire()

        # Check if current block is being read
        while self._remaining_readers[self._writer_index] != 0 and \
                any([reader['reading'] for reader in self._readers.values() if reader['index'] == self._writer_index]):
            self._shared_lock.release()
            time.sleep(0.01)

            if timeout and time.time() - t_start >= timeout:
                return None

            self._shared_lock.acquire()

        # Flag determining whether blob is being overwritten
        overwriting = self._remaining_readers[self._writer_index] > 0

        # Set remaining reader to 0, "clearing" the blob
        self._remaining_readers[self._writer_index] = 0

        # Release shared lock
        self._shared_lock.release()

        # Test to see whether data is being overwritten
        if overwriting:
            logging.warning(f'Overwriting data {threading.current_thread().name} {self._writer_index}')

        # Return data splice
        return self._data[self._writer_index]

    def release_write(self, obs_info):
        """ Finished writing data block, increment writer and release lock
        :param obs_info: Updated observation information """

        # Copy updated observation information to placeholder associated with written block
        if obs_info is not None:
            self._obs_info[self._writer_index] = copy.copy(obs_info)

        # Acquire shared lock
        self._shared_lock.acquire()

        # Set block as written
        self._remaining_readers[self._writer_index] = self._nof_readers

        # Update writer index and release lock
        self._writer_index = (self._writer_index + 1) % self._nof_blocks
        self._shared_lock.release()

    @property
    def shape(self):
        """ Return shape of underlying data """
        return self._shape

    @property
    def datatype(self):
        """ Return the datatype of underlying data """
        return self._data_type

import copy
import logging
import numpy as np
import threading
import time
from multiprocessing import Lock

from pybirales.pipeline.base.definitions import ObservationInfo


class DataBlob(object):
    """ Data blob super class """

    def __init__(self, config, shape, datatype, nof_blocks=3):
        """ Class constructor
        :param config: Blob configuration
        :param shape: The shape of the underlying data type
        :param datatype: Data type of blob data
        :param nof_blocks: Number of data blocks in DataBlob
        """

        # Shape and data type of the data
        self._shape = shape
        self._data_type = datatype

        # The data blob as a numpy array
        data_shape = []
        for item in self._shape:
            data_shape.append(int(item[1]))
        self._data = np.zeros((nof_blocks,) + tuple(data_shape), dtype=datatype)

        # Create observation info object for every data block
        self._obs_info = []

        # Flags to determine whether a block in the blob is written or not
        self._block_has_data = []

        for i in range(nof_blocks):
            self._obs_info.append(ObservationInfo())
            self._block_has_data.append(False)

        # The number of blocks in the data blob
        self._nof_blocks = nof_blocks

        # Lock for reading and current reader index
        self._reader_lock = Lock()
        self._reader_index = 0

        # Lock for writing and current writer index
        self._writer_lock = Lock()
        self._writer_index = 0

    def request_read(self, timeout=None):
        """ Wait while reader and writer are not pointing to the same block, then return read block
        :return: Data splice associated with current block and associated observation information
        """
        t_start = time.time()

        # The reader must wait for the writer to insert data into the blob. When the reader and writer have the
        # same index it means that the reader is waiting for data to be available (the writer must be ahead of reader)
        self._writer_lock.acquire()
        while self._reader_index == self._writer_index and not self._block_has_data[self._reader_index]:
            self._writer_lock.release()
            time.sleep(0.001)  # Wait for data to become available

            # If timeout has elapsed, return None
            if timeout and time.time() - t_start >= timeout:
                return None, None

            self._writer_lock.acquire()

        # Release writer lock and acquire reader lock
        self._writer_lock.release()
        self._reader_lock.acquire()

        # Mark block as being read
        self._block_has_data[self._reader_index] = None

        # Return data splice
        return self._data[self._reader_index, :], copy.copy(self._obs_info[self._reader_index])

    def release_read(self):
        """ Finished reading data block, increment reader index and release lock """
        self._block_has_data[self._reader_index] = False
        self._reader_index = (self._reader_index + 1) % self._nof_blocks
        self._reader_lock.release()

    def request_write(self, timeout=None):
        """ Get next block for writing
        :return: Data splice associated with current block
        """
        t_start = time.time()

        # Acquire writer lock (if data is being read, wait for it to finish)
        self._writer_lock.acquire()
        while self._block_has_data[self._writer_index] is None:
            self._writer_lock.release()
            time.sleep(0.01)

            if timeout and time.time() - t_start >= timeout:
                return None

            self._writer_lock.acquire()

        # Test to see whether data is being overwritten
        if self._block_has_data[self._writer_index]:
            logging.warning('Overwriting data [%s] %d %d',
                            threading.current_thread().name,
                            self._writer_index,
                            self._reader_index)

        # Return data splice
        return self._data[self._writer_index, :]

    def release_write(self, obs_info):
        """ Finished writing data block, increment writer and release lock
        :param obs_info: Updated observation information """

        # Copy updated observation information to placeholder associated with written block
        if obs_info is not None:
            self._obs_info[self._writer_index] = copy.copy(obs_info)

        # Set block as written
        self._block_has_data[self._writer_index] = True

        # Update writer index and release lock
        self._writer_index = (self._writer_index + 1) % self._nof_blocks
        self._writer_lock.release()

    def get_snapshot(self, index=None):
        """ Get a data snapshot from the blob
        :param index: Index of sub-array to send to caller
        :return: snapshot with observation info
        """
        # Assign snapshot index to the previously read blob
        snapshot_index = self._reader_index - 1
        while snapshot_index == self._writer_index and not self._block_has_data[snapshot_index]:
            time.sleep(0.001)

        # Required required segment of data
        to_return = self._data[snapshot_index][index].copy(), copy.copy(self._obs_info[snapshot_index])

        # All done, return data segment and associated obs_info
        return to_return

    @property
    def shape(self):
        """ Return shape of underlying data """
        return self._shape

    @property
    def datatype(self):
        """ Return the datatype of underlying data """
        return self._data_type

import copy
import numpy as np
import time
from multiprocessing import Lock

from pybirales.base.definitions import ObservationInfo


class DataBlob(object):
    """ Data blob super class """

    def __init__(self, config, shape, nof_blocks=4):
        """ Class constructor
        :param shape: The shape of the underlying data type
        :param nof_blocks: Number of data blocks in DataBlob
        """

        # Shape of the data
        self._shape = shape

        # The data blob as a numpy array
        data_shape = []
        for item in self._shape:
            data_shape.append(item[1])
        self._data = np.zeros((nof_blocks,) + tuple(data_shape))

        # Create observation info object for every data block
        self._obs_info = []

        # Flags to determine whether a block in the blob is written or not
        self._block_has_data = []

        for i in xrange(nof_blocks):
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

        # Add items in config to instance
        if config is not None:
            for x, y in config.iteritems():
                self.__dict__[x] = y

    def request_read(self):
        """ Wait while reader and writer are not pointing to the same block, then return read block
        :return: Data splice associated with current block and associated observation information
        """

        # The reader must wait for the writer to insert data into the blob. When the reader and writer have the
        # same index it means that the reader is waiting for data to be available (the writer must be ahead of reader)
        self._writer_lock.acquire()
        while self._reader_index == self._writer_index and not self._block_has_data[self._reader_index]:
            self._writer_lock.release()
            time.sleep(0.001)  # Wait for data to become available
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

    def request_write(self):
        """ Get next block for writing
        :return: Data splice associated with current block
        """

        # Acquire writer lock (if data is being read, wait for it to finish)
        self._writer_lock.acquire()
        while self._block_has_data[self._writer_index] is None:
            self._writer_lock.release()
            time.sleep(0.001)
            self._writer_lock.acquire()

        # Return data splice
        return self._data[self._writer_index, :]

    def release_write(self, obs_info):
        """ Finished writing data block, increment writer and release lock
        :param obs_info: Updated observation information """

        # Copy updated observation information to placeholder associated with written block
        self._obs_info[self._writer_index] = copy.copy(obs_info)

        # Set block as written
        self._block_has_data[self._writer_index] = True

        # Update writer index and release lock
        self._writer_index = (self._writer_index + 1) % self._nof_blocks
        self._writer_lock.release()

    def request_snapshot(self):
        """ Get a data snapshot from the blob
        :return: snapshot with observation info
        """
        # TODO: implement
        pass

    @property
    def shape(self):
        """ Return shape of underlying data """
        return self._shape

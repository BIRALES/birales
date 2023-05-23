import cupy as cu

from pybirales.pipeline.base.blob import DataBlob


class GPUDataBlob(DataBlob):
    """ Blob classes with data located in GPU memory rather than RAM """

    def __init__(self, shape, datatype, nof_blocks=2, device=0):
        """ Class constructor"""

        # First set device where data will be located
        self._device = device

        # Then call super constructor which will call initialise to set up the buffers
        super(GPUDataBlob, self).__init__(shape, datatype, nof_blocks)

    def initialise(self, nof_blocks, datatype):
        # The data blob as a cupy array
        with cu.cuda.Device(self._device):
            data_shape = []
            for item in self._shape:
                data_shape.append(int(item[1]))
            return cu.zeros((nof_blocks,) + tuple(data_shape), dtype=datatype)

from pybirales.pipeline.base.blob import DataBlob
from pybirales.pipeline.base.gpu_blob import GPUDataBlob


class DummyBlob(DataBlob):
    """ DataBlob representing dummy data """

    def __init__(self, shape, datatype, nof_blocks=4):
        """ Class initializer """

        # Call superclass initializer
        super(DummyBlob, self).__init__(shape, datatype, nof_blocks)


class GPUDummyBlob(GPUDataBlob):
    """ DataBlob representing dummy data """

    def __init__(self, shape, datatype, nof_blocks=2, device=0):
        """ Class initializer """

        # Call superclass initializer
        super(GPUDummyBlob, self).__init__(shape, datatype, nof_blocks=nof_blocks, device=device)

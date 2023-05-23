from pybirales.pipeline.base.blob import DataBlob
from pybirales.pipeline.base.gpu_blob import GPUDataBlob


class ReceiverBlob(DataBlob):
    """ DataBlob representing dummy data """

    def __init__(self, shape, datatype, nof_blocks=4):
        """ Class initializer """

        # Call superclass initializer
        super(ReceiverBlob, self).__init__(shape, datatype, nof_blocks=nof_blocks)


class GPUReceiverBlob(GPUDataBlob):
    """ DataBlob representing dummy data """

    def __init__(self, shape, datatype, nof_blocks=2, device=0):
        """ Class initializer """

        # Call superclass initializer
        super(GPUReceiverBlob, self).__init__(shape, datatype, nof_blocks=nof_blocks, device=device)

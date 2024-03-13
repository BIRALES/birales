from pybirales.pipeline.base.blob import DataBlob
from pybirales.pipeline.base.gpu_blob import GPUDataBlob


class ChannelisedBlob(DataBlob):
    """ DataBlob representing channelised data """

    def __init__(self, shape, datatype, nof_blocks=4):
        """ Class initializer """

        # Call superclass initializer
        super(ChannelisedBlob, self).__init__(shape, datatype, nof_blocks=nof_blocks)


class GPUChannelisedBlob(GPUDataBlob):
    """ DataBlob representing channelised data """

    def __init__(self, shape, datatype, nof_blocks=2, device=0):
        """ Class initializer """

        # Call superclass initializer
        super(GPUChannelisedBlob, self).__init__(shape, datatype, nof_blocks=nof_blocks, device=device)
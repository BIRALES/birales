from pybirales.pipeline.base.blob import DataBlob
from pybirales.pipeline.base.gpu_blob import GPUDataBlob


class BeamformedBlob(DataBlob):
    """ DataBlob representing beamformed data """

    def __init__(self, shape, datatype, nof_blocks=4):
        """ Class initializer """

        # Call superclass initializer
        super(BeamformedBlob, self).__init__(shape, datatype, nof_blocks=nof_blocks)


class GPUBeamformedBlob(GPUDataBlob):
    """ DataBlob representing beamformed data """

    def __init__(self, shape, datatype, nof_blocks=2, device=0):
        """ Class initializer """

        # Call superclass initializer
        super(GPUBeamformedBlob, self).__init__(shape, datatype, nof_blocks=nof_blocks, device=device)
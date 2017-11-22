from pybirales.pipeline.base.blob import DataBlob


class BeamformedBlob(DataBlob):
    """ DataBlob representing beamformed data """

    def __init__(self, config, shape, datatype, nof_blocks=4):
        """ Class initialiser
        :param config: Configuration object """

        # Call superclass initialiser
        super(BeamformedBlob, self).__init__(config, shape, datatype, nof_blocks)

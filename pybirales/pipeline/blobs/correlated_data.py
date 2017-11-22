from pybirales.pipeline.base.blob import DataBlob


class CorrelatedBlob(DataBlob):
    """ DataBlob representing channelised data """

    def __init__(self, config, shape, datatype, nof_blocks=4):
        """ Class initialiser
        :param config: Configuration object """

        # Call superclass initialiser
        super(CorrelatedBlob, self).__init__(config, shape, datatype, nof_blocks)
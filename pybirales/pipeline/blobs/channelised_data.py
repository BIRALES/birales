from pybirales.pipeline.base.blob import DataBlob


class ChannelisedBlob(DataBlob):
    """ DataBlob representing channelised data """

    def __init__(self, config, shape, datatype, nof_blocks=4):
        """ Class initializer
        :param config: Configuration object """

        # Call superclass initializer
        super(ChannelisedBlob, self).__init__(config, shape, datatype, nof_blocks)

from pybirales.base.blob import DataBlob


class ChannelisedBlob(DataBlob):
    """ DataBlob representing channelised data """

    def __init__(self, config, shape, nof_blocks=4):
        """ Class initialiser
        :param config: Configuration object """

        # Call superclass initialiser
        super(ChannelisedBlob, self).__init__(config, shape, nof_blocks)
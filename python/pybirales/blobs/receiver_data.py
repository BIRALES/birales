from pybirales.base.blob import DataBlob


class ReceiverBlob(DataBlob):
    """ DataBlob representing dummy data """

    def __init__(self, config, shape, datatype, nof_blocks=4):
        """ Class initialiser
        :param config: Configuration object """

        # Call superclass initialiser
        super(ReceiverBlob, self).__init__(config, shape, datatype, nof_blocks)

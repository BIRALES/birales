from pybirales.base.blob import DataBlob


class DummyBlob(DataBlob):
    """ DataBlob representing dummy data """

    def __init__(self, config, shape, nof_blocks=4):
        """ Class initialiser
        :param config: Configuration object """

        # Call superclass initialiser
        super(DummyBlob, self).__init__(config, shape, nof_blocks)

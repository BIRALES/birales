import logging as log


class DetectorException(Exception):
    pass


class DetectionClustersCouldNotBeComparedException(DetectorException):
    def __init__(self):
        self.msg = "Detection clusters cannot be compared"
        log.exception(self.msg)
        Exception.__init__(self, self.msg)

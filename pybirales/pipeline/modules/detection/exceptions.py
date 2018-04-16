import logging as log


class DetectorException(Exception):
    pass


class DetectionClustersCouldNotBeComparedException(DetectorException):
    def __init__(self, msg="Detection clusters cannot be compared"):
        self.msg = msg
        log.exception(self.msg)
        Exception.__init__(self, self.msg)


class DetectionClusterIsNotValid(DetectorException):
    def __init__(self, cluster, msg="Detection cluster is not valid"):
        self.msg = msg
        log.debug('Cluster {} (n: {}) is not valid'.format(id(cluster), cluster.shape[0]))
        Exception.__init__(self, self.msg)


class NoDetectionClustersFound(DetectorException):
    def __init__(self):
        Exception.__init__(self)

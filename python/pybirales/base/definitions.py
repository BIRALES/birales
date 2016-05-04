import datetime


class ObservationInfo(object):
    """ Placeholder class for observation information """

    def __init__(self):
        self._time = datetime.datetime.now()
        self._nsamp = 0
        self._tsamp = 0


class NotConfiguredError(Exception):
    """ Define an exception which occurs when a data blob which is not configured is used """
    pass


class PipelineError(Exception):
    """ Define an exception which occurs when a pipeline error occurs """
    pass
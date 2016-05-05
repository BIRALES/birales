import collections


class ObservationInfo(collections.MutableMapping):
    """ An Observation Info object which is essentially a dict with
        minor adjustments"""

    def __init__(self, *args, **kwargs):
        # Define internal dictionary
        self.store = dict()
        self.update(dict(*args, **kwargs))

        # Add timestamp information
        self['sampling_time'] = 0.0
        self['timestamp'] = 0.0
        self['nchans'] = 0
        self['nsamp'] = 0

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __contains__(self, x):
        return x in self.store

    def __str__(self):
        return str(self.store)


class NotConfiguredError(Exception):
    """ Define an exception which occurs when a data blob which is not configured is used """
    pass


class PipelineError(Exception):
    """ Define an exception which occurs when a pipeline error occurs """
    pass
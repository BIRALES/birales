import sys

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping

from astropy.time import Time
from pybirales import settings


class ObservationInfo(MutableMapping):
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
        self['configuration_id'] = Time.now().unix
        self['created_at'] = Time.now().iso

        self['settings'] = {a: settings.__dict__[a].__dict__ for a in settings.__dict__.keys() if
                            not a.startswith('__') and settings.__dict__[a] is not None}

    def get_dict(self):
        return self.store

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


class NoDataReaderException(Exception):
    """
    Pipeline has reached the end of the data file
    """
    pass


class InputDataNotValidException(PipelineError):
    """
    Input data passed to a module is not the type that the module is expecting
    """
    pass


class PipelineBuilderIsNotAvailableException(Exception):
    def __init__(self, builder_id, available_builders):
        available_pipelines = ', '.join(available_builders)
        Exception.__init__(self, "The '{}' pipeline is not available. Please choose one from the following: {}"
                           .format(builder_id, available_pipelines))

        self.builder_id = builder_id


class InvalidCalibrationCoefficientsException(PipelineError):
    """
    The calibration coefficients are not valid and can't be used

    """
    pass


class BEST2PointingException(PipelineError):
    """
    Something went wrong whilst pointing the BEST2 Antenna
    """

    pass

class BIRALESObservationException(PipelineError):
    """
    Something went wrong whilst starting the observation
    """

    pass


class ROACHBackendException(PipelineError):
    """
    Something went wrong whilst trying to load the ROACH backend
    """

    pass

class TPMBackendException(PipelineError):
    """
    Something went wrong whilst trying to load the TPM backend
    """

    pass


class CalibrationFailedException(PipelineError):
    """
    Something went wrong whilst trying to calibrate
    """

    pass
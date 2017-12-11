import numpy as np
import logging
import struct
# import numba

from pybirales.pipeline.blobs.correlated_data import CorrelatedBlob
from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob
from pybirales.pipeline.blobs.dummy_data import DummyBlob


# @numba.jit(nopython=True, nogil=True)
def correlate(input_data, output_data, nchans, nants, integrations, nsamp):
    for c in xrange(nchans):
        baseline = 0
        for antenna1 in xrange(nants):
            for antenna2 in xrange(antenna1 + 1, nants):
                for i in xrange(nsamp / integrations):
                    """
                    output_data[i, c, baseline, 0] = \
                       np.dot(input_data[0, c, antenna1,
                              i * integrations:(i + 1) * integrations],
                             np.conj(input_data[0, c, antenna2,
                                      i * integrations:(i + 1) * integrations]))
                    """
                    output_data[i, c, baseline, 0] = np.correlate(
                        input_data[0, c, antenna1, i * integrations:(i + 1) * integrations],
                        input_data[0, c, antenna2, i * integrations:(i + 1) * integrations])
                baseline += 1


class Correlator(ProcessingModule):
    """ Correlator module """

    def __init__(self, config, input_blob=None):

        self._validate_data_blob(input_blob, valid_blobs=[ReceiverBlob, DummyBlob, ChannelisedBlob])

        # Check if we're dealing with channelised data or receiver data
        self._after_channelizer = True if type(input_blob) is ChannelisedBlob else False

        # Sanity checks on configuration
        if {'integrations'} - set(config.settings()) != set():
            raise PipelineError("Correlator: Missing keys on configuration. (integrations)")

        # Call superclass initialiser
        super(Correlator, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Correlator"

        # Populate variables
        self._integrations = config.integrations

        # Define parameters
        self._current_input = None
        self._current_output = None
        self._nbaselines = None
        self._nstokes = None
        self._nsamp = None
        self._nchans = None
        self._nants = None
        self._npols = None

    def generate_output_blob(self):
        """ Generate output data blob """
        input_shape = dict(self._input.shape)
        datatype = self._input.datatype

        baselines = int(0.5 * ((input_shape['nants'] ** 2) - input_shape['nants']))
        nstokes = input_shape['npols'] ** 2

        # Check if integration time is a multiple of nsamp
        if input_shape['nsamp'] % self._config.integrations != 0:
            raise PipelineError("Correlator: integration time must be a multiple of nsamp")

        # Generate output blob
        return CorrelatedBlob(self._config, [('nsamp', input_shape['nsamp'] / self._config.integrations),
                                             ('nchans',
                                              input_shape['nchans'] if self._after_channelizer else input_shape[
                                                  'nsubs']),
                                             ('baselines', baselines),
                                             ('stokes', nstokes)],
                              datatype=datatype)

    def process(self, obs_info, input_data, output_data):
        """ Perform correlation """

        # Update parameters
        self._nsamp = obs_info['nsamp']
        self._nchans = obs_info['nsubs']
        self._nants = obs_info['nants']
        self._npols = obs_info['npols']

        self._nbaselines = int(0.5 * ((obs_info['nants'] ** 2) - obs_info['nants']))
        self._nstokes = obs_info['npols'] ** 2

        # Integration check 
        if self._nsamp % self._integrations != 0:
            logging.warning("Number of integration not factor of number of samples, skipping buffer")
            return

        calib_coeffs = np.array([1 + 0j,
                                 0.74462 - 0.73813j,
                                 0.96279 - 0.16081j,
                                 1.219 - 0.134j,
                                 1.1013 + 0.11142j,
                                 1.2633 + 0.36585j,
                                 0.68739 - 0.89354j,
                                 0.73833 - 0.9012j,
                                 1.0109 - 0.1333j,
                                 0.81491 - 0.56862j,
                                 1.0063 - 0.22017j,
                                 0.23828 - 0.96069j,
                                 0.69746 - 0.61241j,
                                 -0.35428 + 0.93366j,
                                 0.81765 - 0.64325j,
                                 0.31159 - 1.1006j,
                                 -0.78607 + 0.64746j,
                                 0.5285 + 0.90175j,
                                 0.30446 + 0.922j,
                                 0.018304 + 1.038j,
                                 0.11882 + 1.0397j,
                                 0.079158 + 0.99265j,
                                 -0.0089506 + 1.0346j,
                                 0.25724 + 1.1163j,
                                 -0.46037 + 1.0724j,
                                 0.88436 + 0.55704j,
                                 1.094 - 0.16808j,
                                 1.0345 + 0.36037j,
                                 0.67881 + 0.64781j,
                                 -0.67647 + 0.79722j,
                                 0.97867 - 0.045864j,
                                 1.0432 - 0.16661j], dtype=np.complex64)

        # Apply coeffs
        #	input_data *= calib_coeffs

        # Transpose the data so the we can parallelise over frequency
        self._current_input = np.transpose(input_data, (0, 1, 3, 2)).copy()

        # Perform correlation
        if self._npols == 1:
            correlate(self._current_input, output_data, self._nchans,
                      self._nants, self._integrations, self._nsamp)

        obs_info['nsamp'] = self._nsamp / self._integrations
        obs_info['sampling_time'] *= self._integrations
        obs_info['nbaselines'] = self._nbaselines
        obs_info['nstokes'] = self._nstokes

        return obs_info

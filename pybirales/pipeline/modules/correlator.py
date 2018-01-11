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

        calib_coeffs = np.array([1.00000000 +0.00000000e+00j,  1.00000000 -5.10307927e-07j,
  1.00000000 -1.02061585e-06j,  1.00000000 -1.53087865e-06j,
 -0.81171221 +5.84057629e-01j, -0.81171191 +5.84058046e-01j,
 -0.81171161 +5.84058464e-01j, -0.81171131 +5.84058881e-01j,
  0.31775340 -9.48173404e-01j,  0.31775290 -9.48173583e-01j,
  0.31775242 -9.48173702e-01j,  0.31775194 -9.48173881e-01j,
  0.29586360 +9.55230176e-01j,  0.29586408 +9.55230057e-01j,
  0.29586455 +9.55229878e-01j,  0.29586506 +9.55229759e-01j,
 -0.79806554 -6.02570653e-01j, -0.79806584 -6.02570236e-01j,
 -0.79806620 -6.02569818e-01j, -0.79806650 -6.02569401e-01j,
  0.99973553 +2.29976550e-02j,  0.99973553 +2.29971446e-02j,
  0.99973553 +2.29966342e-02j,  0.99973553 +2.29961239e-02j,
 -0.82492948 +5.65235674e-01j, -0.82492918 +5.65236092e-01j,
 -0.82492888 +5.65236509e-01j, -0.82492858 +5.65236926e-01j,
  0.33947513 -9.40615058e-01j,  0.33947465 -9.40615237e-01j,
  0.33947417 -9.40615356e-01j,  0.33947366 -9.40615535e-01j], dtype=np.complex)

        # Apply coeffs
        input_data *= calib_coeffs

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

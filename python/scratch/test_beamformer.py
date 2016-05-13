
from numpy import ctypeslib
import numpy as np
import ctypes
import time

# Load library
library = ctypes.CDLL("libbeamformer.so")

# Define setReceiverConfiguration function
complex_p = ctypeslib.ndpointer(np.complex64, ndim=1, flags='C')
library.beamformer.argtypes = [complex_p, complex_p, complex_p, ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
library.beamformer.restype = None


def beamform(data, weights, output):
    """ Apply pointing weights to data
    :param data: Data in nchans/nsamp/nant order
    :param weights: Weights in nchans/nants/nbeams order
    :param output: Output in nbeams/nchans/nsamp order
    """
    library.beamformer(data.ravel(), data.ravel(), output.ravel(),
                       nsamp, nchans, nbeams, nants, nthreads)

if __name__ == "__main__":

    nsamp = 1024 * 1024 * 4
    nchans = 1
    nants = 32
    nbeams = 32
    nthreads = 8

    input_data = np.zeros((nchans, nsamp, nants), dtype=np.complex64) + 1+1j
    weights = np.zeros((nchans, nbeams, nants), dtype=np.complex64) + 1+1j
    output = np.zeros((nbeams, nchans, nsamp), dtype=np.complex64)

    tstart = time.time()
    beamform(input_data, weights, output)
    tend = time.time()


    print "Run time (serial):\t\t %f" % (tend - tstart)

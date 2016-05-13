import math
import numpy as np
import time

from multiprocessing.pool import ThreadPool
import numba
from matplotlib import pyplot as plt
import random


@numba.jit(nopython=True, nogil=True)
def apply_fir_filter(input, fir_filter, output, ntaps, nchans):
    nof_samples = len(input)
    for n in xrange(ntaps * nchans, nof_samples):
        m = n % nchans
        coeff_sub = fir_filter[nchans * ntaps - m:: -nchans]
        output[m, (n - ntaps * nchans) / nchans] = (input[n - (ntaps - 1) * nchans: n + nchans: nchans] *
                                                    coeff_sub).sum()


@numba.jit(nopython=True, nogil=True)
def apply_fir_filter_fast(data, fir_filter, output, ntaps, nchans):
    nof_spectra = (len(data) - ntaps * nchans) / nchans
    for n in xrange(nof_spectra):
        temp = data[n * nchans: n * nchans + nchans * ntaps] * fir_filter
        for j in xrange(1, ntaps):
            temp[:nchans] += temp[j * nchans: (j + 1) * nchans]
        output[:, n] = temp[:nchans]


class PFB(object):
    def __init__(self, nchans, ntaps, nsamp, nbeams, nthreads=4):
        """ Class constructor
        :param nchans: Number of channels to generate
        :param ntaps: Number of filter taps
        :param nsamp: Number of samples per block
        :param nbeams: Number of beams in input data
        :param nthreads: Number of threads to use
        :return:
        """
        self._ntaps = ntaps
        self._nchans = nchans
        self._nbeams = nbeams
        self._bin_width_scale = 1.0
        self._filter = None

        # Generate FIR filter
        self._generate_filter()

        # Create thread pool
        self._nthreads = nthreads
        self._thread_pool = ThreadPool(nthreads)

        # Data containers
        self._input = None
        self._filtered = np.zeros((nbeams, nchans, (nsamp - ntaps * nchans) / nchans), dtype=np.complex64)
        self._output = np.zeros((nbeams, self._nchans, (nsamp - ntaps * nchans) / self._nchans), dtype=np.complex64)

    def _generate_filter(self):
        """ Generate FIR filter """
        dx = math.pi / self._nchans
        X = np.array([n * dx - self._ntaps * math.pi / 2 for n in range(self._ntaps * self._nchans)])
        self._filter = np.sinc(self._bin_width_scale * X / math.pi) * np.hanning(self._ntaps * self._nchans)

        # Reverese filter to ease fast computation
        self._filter = self._filter[::-1]

    def channelise(self):
        """ Perform channelisation, serial version """
        for b in range(self._nbeams):
            apply_fir_filter_fast(self._input[b, :], self._filter, self._filtered[b, :], self._ntaps, self._nchans)
            self._output[b, :] = np.abs(np.fft.fft(self._filtered[b, :], axis=0))
        return self._output

    def channelise_thread(self, beam):
        """ Perform channelisation, to be used with ThreadPool
        :param beam: Beam number associated with call  """
        apply_fir_filter_fast(self._input[beam, :], self._filter, self._filtered[beam, :], self._ntaps, self._nchans)
        self._output[beam, :] = np.abs(np.fft.fft(self._filtered[beam, :], axis=0))

    def channelise_parallel(self):
        """ Perform channelisation, parallel version """
        self._thread_pool.map(self.channelise_thread, range(self._nbeams))
        return self._output

    def set_input(self, input_data):
        """ Set input data to channelise
        :param input_data: Input array """
        self._input = input_data

if __name__ == "__main__":

    nsamp = 1024*1024*4
    nbeams = 32
    ntimes = 1

    a = np.zeros((nbeams, nsamp), dtype=np.complex64)
    # for i in xrange(nsamp):
    #     a[0, i] = np.sin(i * 0.1) + 5 * np.sin(i * 0.5) + random.random() * 10

    pfb = PFB(512, 4, nsamp, nbeams, nthreads=4)

    tstart = time.time()
    pfb.set_input(a)
    for x in range(ntimes):
        spectra1 = pfb.channelise().copy()
    tend = time.time()

    tstart_p = time.time()
    pfb.set_input(a)
    for x in range(ntimes):
        spectra2 = pfb.channelise_parallel()
    tend_p = time.time()

    print "Average Run time (serial):\t\t %f" % ((tend - tstart) / ntimes)
    print "Average Run time (parallel):\t %f" % ((tend_p - tstart_p) / ntimes)
    print "Improvement: %.2f %%" % (100.0 - (((tend_p - tstart_p) / (tend - tstart)) * 100.0))

    #for b in xrange(nbeams):
    # plt.imshow(np.abs(spectra2[0, :]), aspect='auto')
    # plt.colorbar()
    # plt.show()


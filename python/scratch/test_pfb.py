import math
import numpy as np
import time

import numba
from matplotlib import pyplot as plt

nsamp = 78125 * 32

class PFB(object):
    def __init__(self, nchans, ntaps):
        self._ntaps = ntaps
        self._nchans = nchans
        self._bin_width_scale = 1.0
        self._filter = None

        self._generate_filter()

    def _generate_filter(self):
        dx = math.pi / self._nchans
        X = np.array([n * dx-self._ntaps*math.pi / 2 for n in range(self._ntaps*self._nchans)])
        self._filter = np.sinc(self._bin_width_scale * X / math.pi) * np.hanning(self._ntaps * self._nchans)

    @numba.jit(nopython=True)
    def _apply_filter(self, x):
        n = len(x)
        y = np.zeros(((n - self._ntaps * self._nchans), self._ntaps), dtype=np.complex64)
        for n in range((self._ntaps - 1) * self._nchans, n):
            m = n % self._nchans
            coeff_sub = self._filter[self._nchans * self._ntaps - m:: -self._nchans]
            y[n - self._ntaps * self._nchans, :] = (x[n - (self._ntaps - 1) * self._nchans: n + self._nchans:self._nchans]
                                                    * coeff_sub)
        y = y.sum(axis=1)
        return y

    @numba.jit(nopython=True)
    def channelise(self, x):
        b = self._apply_filter(x)
        spectra = np.zeros((self._nchans, len(b) / self._nchans))
        for i in xrange(len(b) / self._nchans):
            spectra[:, i] = np.abs(np.fft.fft(b[i * self._nchans: (i+1)*self._nchans]))
        return spectra

if __name__ == "__main__":

    a = np.zeros(nsamp, dtype=np.complex64)
#    for i in xrange(nsamp):
#        a[i] = np.sin(i * 0.1) + 5 * np.sin(i * 0.5) + random.random() * 10

    pfb = PFB(16384, 4)

    tstart = time.time()
    spectra = pfb.channelise(a)
    tend = time.time()
    print "Run time: %f" % (tend - tstart)

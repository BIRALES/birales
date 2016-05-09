from matplotlib import pyplot as plt
import math, random
import numpy as np

nfft = 512

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def pfb_fir(x):
    N = len(x)    # x is the incoming data time stream.
    taps = 4
    L = nfft   # Points in subsequent FFT.
    bin_width_scale = 1.0
    dx = math.pi/L
    X = np.array([n*dx-taps*math.pi/2 for n in range(taps*L)])
    coeff = np.sinc(bin_width_scale*X/math.pi)*np.hanning(taps*L)

    y = np.array([0+0j]*(N-taps*L))
    for n in range((taps-1)*L, N):
        m = n % L
        coeff_sub = coeff[L*taps-m::-L]
        y[n-taps*L] = (x[n-(taps-1)*L:n+L:L]*coeff_sub).sum()

    return y

def pfb_fir_new(x):
    N = len(x)    # x is the incoming data time stream.
    taps = 4
    L = nfft   # Points in subsequent FFT.
    bin_width_scale = 1.0
    dx = math.pi/L
    X = np.array([n*dx-taps*math.pi/2 for n in range(taps*L)])
    coeff = np.sinc(bin_width_scale*X/math.pi)*np.hanning(taps*L)[::-1]

    y = np.array([0+0j]*(N-taps*L))
    nof_spectra = (N - taps * L) / L
    for n in xrange(nof_spectra):
        y[n * L: (n+1)*L] = (x[n * L: n * L + L * taps] * coeff).reshape((taps, L)).sum(axis=0)
    return y

a = np.zeros(16384)
for i in xrange(16384):
   a[i] = np.sin(i * 0.1) + 5 * np.sin(i * 0.5) + random.random() * 10

b = pfb_fir_new(a).tolist()
c = pfb_fir(a).tolist()

spectra1 = np.zeros((nfft, len(b) / nfft))
for i in range(len(b) / nfft):
    spectra1[:,i] = np.abs(np.fft.fft(b[i * nfft : (i+1)*nfft]))

spectra2 = np.zeros((nfft, len(c) / nfft))
for i in range(len(c) / nfft):
    spectra2[:,i] = np.abs(np.fft.fft(c[i * nfft : (i+1)*nfft]))

print spectra1 == spectra2

plt.figure()
plt.imshow(spectra1, aspect='auto', interpolation='none')
plt.colorbar()
plt.figure()
plt.imshow(spectra2, aspect='auto', interpolation='none')
plt.colorbar()
plt.show()
import os
import sys
from offline_pointing import Pointing, config
import numpy as np
import numba


@numba.jit(nopython=True, nogil=True)
def beamformer(i, nbeams, data, weights, coeffs, output):
    for b in range(nbeams):
        output[b, i] = 10 * np.log10(np.sum(np.power(np.abs(np.dot(data * coeffs, weights[0, b, :])), 2)))


filepath = "/mnt/2017/27_06_2017/casa/casa_raw.dat"

nsamp = 32768
nants = 32
start_ra = -0
stop_ra = 1
delta_ra = 1
start_dec = -0
stop_dec = 1
delta_dec = 1

# Update pointing config
config['reference_antenna_location'] = [11.6459889, 44.52357778]
config['reference_declination'] = 58.92

# Generate pointings
config['pointings'] = []
for i in np.arange(start_ra, stop_ra, delta_ra):
    for j in np.arange(start_dec, stop_dec, delta_dec):
        config['pointings'].append([i, j])

config['nbeams'] = len(config['pointings'])

# Create pointing object
pointing = Pointing(config, 1, 32)

# Generate pointings
weights = pointing.weights

# Check filesize
filesize = os.path.getsize(filepath)
totalsamp = filesize / (8 * nants)

# Create output array
output = np.zeros((config['nbeams'], int(totalsamp / nsamp)), dtype=np.float)

calib_coeffs = np.array([1+0j,
1.1181-0.66994j,
0.91954-0.13299j,
0.94726-0.18677j,
0.70255-0.62481j,
1.1135-0.44208j,
0.042972-1.0638j,
0.19723-1.0855j,
0.25116-0.95616j,
-0.17402-0.91246j,
0.14082-0.89012j,
-0.84676-0.59564j,
-0.96256-0.32477j,
0.96903-0.16857j,
-1.0874-0.56989j,
0+0j,
0.99513-0.24908j,
-0.051223-0.93473j,
0.096611-0.96612j,
0.4442-0.89073j,
-0.51102-0.85209j,
-0.27417-0.96805j,
-0.10254-1.0064j,
-0.21471-0.90289j,
-0.6638-0.86633j,
-0.88048+0.19947j,
-0.70341+0.71836j,
-1.0133+0.23887j,
-0.69349+0.66405j,
-0.61236-0.69497j,
-0.15895+0.96957j,
-0.039497+1.0068j], dtype=np.complex64)

# Open file
with open(filepath, 'rb') as f:
    for i in range(int(totalsamp / nsamp)):
        data = f.read(nsamp * nants * 8)
        data = np.frombuffer(data, np.complex64)
        data = data.reshape((nsamp, nants))

        # Perform calibration and beamforming
        beamformer(i, config['nbeams'], data, weights, calib_coeffs, output)
     #   for b in range(config['nbeams']):
     #       output[b, i] = 10 * np.log10(np.sum(np.power(np.abs(np.sum(data * calib_coeffs, axis=1)), 2)))


        sys.stdout.write("Processing %d of %d [%.2f%%]   \r" % (i,
                                                                totalsamp / nsamp,
                                                                (i / float(totalsamp / nsamp) * 100)))
        sys.stdout.flush()

# Save file
np.save("casa_raw_processed", output)

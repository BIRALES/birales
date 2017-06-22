import os
import sys
from offline_pointing import Pointing, config
import numpy as np
import numba


@numba.jit(nopython=True, nogil=True)
def beamformer(i, nbeams, data, weights, coeffs, output):
    for b in range(nbeams):
        output[b, i] = 10 * np.log10(np.sum(np.power(np.abs(np.dot((data * coeffs)[:, :4], weights[0, b, :4])), 2)))


filepath = "/media/lessju/Data/Birales/13_04_2017/casa_raw.dat"

nsamp = 32768
nants = 32
start_ra = -5
stop_ra = 5
delta_ra = 1
start_dec = -5
stop_dec = 5
delta_dec = 1

# Update pointing config
config['reference_antenna_location'] = [11.6459889, 44.52357778]

# Generate pointings
config['pointings'] = []
for i in np.arange(start_ra, stop_ra, delta_ra):
    for j in np.arange(start_dec, stop_dec, delta_dec):
        config['pointings'].append([i, j])

print len(np.arange(start_ra, stop_ra, delta_ra))

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

calib_coeffs = np.array([1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j,
                         1.0 + 0.0j, ], dtype=np.complex64)

# TODO: Check whether this is required
# calib_coeffs.imag = np.deg2rad(calib_coeffs.imag)

# Open file
with open(filepath, 'rb') as f:
    for i in range(int(totalsamp / nsamp)):
        data = f.read(nsamp * nants * 8)
        data = np.frombuffer(data, np.complex64)
        data = data.reshape((nsamp, nants))

        # Perform calibration and beamforming
        beamformer(i, config['nbeams'], data, weights, calib_coeffs, output)
        sys.stdout.write("Processing %d of %d [%.2f%%]   \r" % (i,
                                                                totalsamp / nsamp,
                                                                (i / float(totalsamp / nsamp) * 100)))
        sys.stdout.flush()

# Save file
np.save("casa_raw_processed_test", output)

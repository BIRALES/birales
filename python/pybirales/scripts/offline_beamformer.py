import os
import sys
from offline_pointing import Pointing, config
import numpy as np
import numba


@numba.jit(nopython=True, nogil=True)
def beamformer(i, nbeams, data, weights, coeffs, output):
    for b in range(nbeams):
        output[b, i] = 10 * np.log10(np.sum(np.power(np.abs(np.dot(data * coeffs, weights[0, b, :])), 2)))


filepath = "/mnt/2017/06_06_2017/cygnus/cygnus_raw.dat"

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
config['reference_declination'] = 40.781765

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
0.74462-0.73813j,
0.96279-0.16081j,
1.219-0.134j,
0.93059-0.59947j,
1.2159-0.50152j,
-0.019579-1.1272j,
0.015478-1.1649j,
0.095198-1.0152j,
-0.37284-0.92108j,
0.0094763-1.03j,
-0.88349-0.44626j,
-0.85426-0.36294j,
0.99494-0.085607j,
-0.93417-0.45786j,
-1.1268+0.19645j,
0.98924-0.24188j,
-0.084482-1.0418j,
0.12612-0.96274j,
0.43426-0.94296j,
-0.3467-0.98741j,
-0.29758-0.9503j,
-0.22098-1.0108j,
-0.49866-1.0313j,
-0.5508-1.0289j,
-0.98711+0.34351j,
-0.5513+0.95981j,
-0.92713+0.58359j,
-0.64829+0.67835j,
-0.79674+0.67704j,
0.04517-0.9787j,
0.16587-1.0433j], dtype=np.complex64)

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
np.save("cygnus_raw_processed", output)

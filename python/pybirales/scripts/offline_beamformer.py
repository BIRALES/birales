import os
import sys
from offline_pointing import Pointing, config
import numpy as np
import datetime
import numba


@numba.jit(nopython=True, nogil=True)
def beamformer(i, nbeams, data, weights, output):
    for b in range(nbeams):
        output[b, i] = 10 * np.log10(np.sum(np.power(np.abs(np.dot(data[:,:4], weights[0, b, :4])), 2)))

filepath = "/media/lessju/Data/Birales/13_04_2017/casa_raw.dat"
pointing_time = datetime.datetime(2017, 4, 13, 9, 10, 36)

nsamp = 32768
nants = 32
start_ra = -10
stop_ra = 10.5
delta_ra = 0.5
start_dec = -10
stop_dec = 10.5
delta_dec = 0.5

# Update pointing config
config['reference_antenna_location'] = [44.52357778, 11.6459889]
config['reference_pointing'] = [0, 58.905726]

config['pointings'] = []

for i in np.arange(start_ra, stop_ra, delta_ra):
    for j in np.arange(start_dec, stop_dec, delta_dec):
        config['pointings'].append([i, j])

config['nbeams'] = len(config['pointings'])

# Create pointing object
pointing = Pointing(config, 1, 32)

# Generate pointings
pointing.run(pointing_time)
weights = pointing.weights

# Check filesize
filesize = os.path.getsize(filepath)
totalsamp = filesize / (8 * nants)

# Create output array
output = np.zeros((config['nbeams'], int(totalsamp / nsamp)), dtype=np.float)

# Open file
with open(filepath, 'rb') as f:
    for i in range(int(totalsamp / nsamp)):
        data = f.read(nsamp * nants * 8)
        data = np.frombuffer(data, np.complex64)
        data = data.reshape((nsamp, nants))

        # Perform beamforming
        beamformer(i, config['nbeams'], data, weights, output)

        sys.stdout.write("Processing %d of %d [%.2f%%]   \r" % (i,
                                                                totalsamp / nsamp,
                                                                (i / float(totalsamp / nsamp) * 100)))
        sys.stdout.flush()

# Save file
np.save("casa_raw_processed_one_cylinder", output)

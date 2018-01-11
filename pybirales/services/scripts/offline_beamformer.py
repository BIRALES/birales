import os
import sys
from offline_pointing import Pointing, config
import numpy as np


def beamformer(i, nbeams, data, weights, coeffs, output):
    for b in range(nbeams):
        x = np.dot(data, weights[0, b, :])
        output[b, i] = np.sum(np.power(np.abs(x), 2))


filepath = "/mnt/2017/20_12_2017/casa/2017_12_20/CasA/CasA_raw.dat"

nsamp = 32768
nants = 32

# Update pointing config
config['reference_antenna_location'] = [11.6459889, 44.52357778]
config['reference_declination'] = 57.917574

# Generate pointings
config['pointings'] = [[-1.6, 1], [0, 1], [1.6, 1], [-1.6, 0], [0, 0], [1.6, 0], [-1.6, -1], [0, -1], [1.6, -1]]
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

# Open file
with open(filepath, 'rb') as f:
    for i in range(int(totalsamp / nsamp)):
        data = f.read(nsamp * nants * 8)
        data = np.frombuffer(data, np.complex64)
        data = data.reshape((nsamp, nants))

        # Perform calibration and beamforming
        beamformer(i, config['nbeams'], data, weights, None, output)
        sys.stdout.write("Processing %d of %d [%.2f%%]   \r" % (i,
                                                                totalsamp / nsamp,
                                                                (i / float(totalsamp / nsamp) * 100)))
        sys.stdout.flush()

# Save file
np.save("casa_raw_processed_offset", output)

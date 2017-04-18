import os
import sys
from offline_pointing import Pointing, config
import numpy as np

filepath = "/mnt/2017/13_04_2017/casa/casa_raw.dat"
nsamp = 8192
nants = 32

# Update pointing config
config['reference_antenna_location'] = [44.52357778, 11.6459889]
config['reference_pointing'] = [0, 64.515]

config['pointings'] = []
for i in range(-100, 100):
    for j in range(-100, 100):
        config['pointings'].append([i * 0.2, j * 0.2])

config['nbeams'] = len(config['pointings'])

# Create pointing object
pointing = Pointing(config, 1, 32)

# Check filesize
filesize = os.path.getsize(filepath)
totalsamp = filesize / (8 * nants)

# Create output array
output = np.zeros((len(config['nbeams']), (totalsamp / nsamp)), dtype=np.float)
print(output.shape)

# Open file
with open(filepath, 'rb') as f:
    for i in range(totalsamp/nsamp):
        data = f.read(nsamp * nants * 8)
        data = np.frombuffer(data, np.complex64)
        data = data.reshape((nsamp, nants))

        output[i] = 10 * np.log10(np.sum(np.power(np.abs(np.dot(data, weights)), 2)))

        sys.stdout.write("Processing %d of %d [%.2f%%]   \r" % (i,
                                                                totalsamp/nsamp,
                                                                (i / float(totalsamp/nsamp) * 100)))
        sys.stdout.flush()



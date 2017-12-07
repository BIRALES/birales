import os
import sys
from offline_pointing import Pointing, config
import numpy as np
import numba


#@numba.jit(nopython=True, nogil=True)
def beamformer(i, nbeams, data, weights, coeffs, output):
    for b in range(nbeams):
        x = np.dot(data * coeffs, weights[0, b, :])
        output[b, i] = np.sum(np.power(np.abs(x), 2))


filepath = "/mnt/2017/06_12_2017/virgo/virgo_raw.dat"

nsamp = 32768 * 4
nants = 32
start_ra = -0
stop_ra = 1
delta_ra = 1
start_dec = -0
stop_dec = 1
delta_dec = 1

# Update pointing config
config['reference_antenna_location'] = [11.6459889, 44.52357778]
config['reference_declination'] = 12.293301

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

masks = {
'virgo_1_antenna': np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
'virgo_2_antenna': np.array([0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
'virgo_cylinder_1': np.array([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
'virgo_cylinder_2': np.array([0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
'virgo_cylinder_3': np.array([0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
'virgo_cylinder_4': np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
'virgo_cylinder_5': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]),
'virgo_cylinder_6': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0]),
'virgo_cylinder_7': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0]),
'virgo_cylinder_8': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]),
'virgo_2_cylinders': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]),
'virgo_4_cylinders': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]),
'virgo_full_array': np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
}

for key, mask in masks.iteritems():
    print "Processing", key
    # Open file
    output[:] = 0
    with open(filepath, 'rb') as f:
        for i in range(int(totalsamp / nsamp)):
            data = f.read(nsamp * nants * 8)
            data = np.frombuffer(data, np.complex64)
            data = data.reshape((nsamp, nants))

            # Perform calibration and beamforming
            beamformer(i, config['nbeams'], data, weights, mask, output)
            sys.stdout.write("Processing %d of %d [%.2f%%]   \r" % (i,
                                                                    totalsamp / nsamp,
                                                                    (i / float(totalsamp / nsamp) * 100)))
            sys.stdout.flush()

    # Save file
    # np.save("casa_raw_processed", output)

    with open("{}.txt".format(key), 'w') as r:
        [r.write(str(x[0])+'\n') for x in output.T]

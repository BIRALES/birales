import os
import sys
from offline_pointing import Pointing, config
import numpy as np
import numba


@numba.jit(nopython=True, nogil=True)
def beamformer(i, nbeams, data, weights, coeffs, output):
    for b in range(nbeams):
        output[b, i] = 10 * np.log10(np.sum(np.power(np.abs(np.dot(data * coeffs, weights[0, b, :])), 2)))


filepath = "/mnt/2017/06_06_2017/casa/casa_raw.dat"

nsamp = 32768
nants = 32
start_ra = -10
stop_ra = 0.5
delta_ra = 10
start_dec = -10
stop_dec = 0.2
delta_dec = 10

# Update pointing config
config['reference_antenna_location'] = [11.6459889, 44.52357778]
config['reference_declination'] = 58.905452

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

calib_coeffs = np.array([1+0j,
0.74462-0.73813j,
0.96279-0.16081j,
1.219-0.134j,
-0.32421-1.0584j,
-0.15246-1.3064j,
-1.0901-0.28728j,
-1.1169-0.33127j,
-0.61186+0.81568j,
-0.16366+0.98011j,
-0.54654+0.87314j,
0.51988+0.84227j,
0.87381-0.31297j,
-0.67283+0.73793j,
0.99691-0.29745j,
0.69452-0.90883j,
0.66416+0.77201j,
0.89022-0.5477j,
0.91526-0.32415j,
1.0374-0.040547j,
-0.532-0.9012j,
-0.4766-0.87434j,
-0.41321-0.94853j,
-0.68958-0.9147j,
-1.0694+0.46741j,
0.26409+1.0113j,
0.91304+0.62572j,
0.50817+0.97052j,
0.84344-0.41115j,
0.51221+0.91149j,
0.29735-0.93353j,
0.20658-1.036j], dtype=np.complex64)

#calib_coeffs = np.array([1+0j,
#        0.37921-1.0166j,
#        1.0541+0.011449j,
#        1.1097-0.21978j,
#        -0.53587-0.61423j,
#        -0.68138-0.97568j,
#        -1.0463+0.24731j,
#        -1.4677+0.24342j,
#        0.28354+1.0108j,
#        0.62238+0.71357j,
#        0.44854+1.2109j,
#        1.0746+0.22425j,
#        -0.75818-1.548j,
#        0.61083+0.94167j,
#        -0.42345-1.4099j,
#        -0.83914-0.5053j,
#        0.16475-1.3879j,
#        -1.3017-0.28034j,
#        -0.74333-0.46589j,
#        -0.83767-1.1915j,
#        0.83129+0.65413j,
#        0.23431+0.85324j,
#        -0.0012562+1.120j,
#        -0.070134+1.3493j,
#        0.47897-0.75327j,
#        -0.16405-1.1344j,
#        -0.88662-0.69923j,
#        -0.24001-1.0737j,
#        -0.041437+1.3206j,
#        -1.005-0.90608j,
#        0.19415+0.99563j,
#        0.23096+1.1082j], dtype=np.complex64)


# Open file
with open(filepath, 'rb') as f:
    for i in range(int(totalsamp / nsamp)):
        data = f.read(nsamp * nants * 8)
        data = np.frombuffer(data, np.complex64)
        data = data.reshape((nsamp, nants))

        # Perform calibration and beamforming
        beamformer(i, config['nbeams'], data, weights, calib_coeffs, output)
    #    for b in range(config['nbeams']):
     #       output[b, i] = 10 * np.log10(np.sum(np.power(np.abs(np.sum(data * calib_coeffs, axis=1)), 2)))


        sys.stdout.write("Processing %d of %d [%.2f%%]   \r" % (i,
                                                                totalsamp / nsamp,
                                                                (i / float(totalsamp / nsamp) * 100)))
        sys.stdout.flush()

# Save file
np.save("casa_raw_processed_multipixel", output)

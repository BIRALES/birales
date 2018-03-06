import os
import sys
from offline_pointing import Pointing, config
import numpy as np


def beamformer(i, nbeams, data, weights, coeffs, output):
    for b in range(nbeams):
        x = np.dot(data, weights[0, b, :])
        output[b, i] = np.sum(np.power(np.abs(x), 2))


# filepath = "/mnt/2018_02_22/FesCas4/FesCas4_raw.dat"
filepath = "/mnt/2018_02_28/FesVirgo1/FesVirgo1_raw.dat"

obs = 'Vir2802'
cal = 'Vir2802'

nsamp = 32768
nants = 32
skip = 64

# Update pointing config
config['reference_antenna_location'] = [11.6459889, 44.52357778]
config['reference_declination'] = 12

# Generate pointings
config['pointings'] = [[0, 0]]
config['nbeams'] = len(config['pointings'])

# Create pointing object
pointing = Pointing(config, 1, 32)
#print pointing.weights

# Generate pointings
weights = pointing.weights

# Check filesize
filesize = os.path.getsize(filepath)
totalsamp = filesize / (8 * nants)

# Create output array
output = np.zeros((config['nbeams'], int(totalsamp / nsamp) / skip), dtype=np.float)

calib_coeffs = np.array([1.000000+0.000000j,
1.077067-1.272407j,
1.209792-0.707701j,
1.094349-1.111881j,
1.253441-0.829843j,
-0.658861-1.877637j,
1.986287+0.451275j,
-0.222136-1.844457j,
1.030176+1.133961j,
1.594387+0.327530j,
0.272793+1.490776j,
0.391006-1.486456j,
1.451742+0.425590j,
1.238944+0.948830j,
1.449291-0.754724j,
0.918053+0.396458j,
-1.094343+1.415086j,
0.044612+1.782410j,
-0.798266+1.688173j,
1.069074+1.054116j,
-0.975894+1.170727j,
-1.635989+0.192003j,
-0.695983+1.450412j,
-1.022384+1.645008j,
-1.446848+1.290698j,
-1.721431+0.087368j,
0.058489+1.707038j,
0.490243+1.811800j,
1.105667+1.335101j,
0.293915+1.774871j,
-1.628916+0.521056j,
-1.424269-0.574465j], dtype=np.complex64)

#weights = np.ones((1, 1, 32), dtype=np.complex64)
#weights_real = np.ones(len(calib_coeffs))
#weights_imag = np.ones(len(calib_coeffs))
#for i in range(len(calib_coeffs)):
#	weights_real[i] = calib_coeffs[i].real * weights[0, 0, i].real
#	weights_imag[i] = calib_coeffs[i].imag * weights[0, 0, i].imag
#	weights[0, 0, i] = np.complex(weights_real[i], weights_imag[i])
weights = calib_coeffs * weights
print weights

# Open file
with open(filepath, 'rb') as f:
    for i in range(0, int(totalsamp / nsamp) / skip):
        f.seek(nsamp * nants * 8 * i * skip, 0)
        data = f.read(nsamp * nants * 8)
        data = np.frombuffer(data, np.complex64)
        data = data.reshape((nsamp, nants))

        # Perform calibration and beamforming
        beamformer(i, config['nbeams'], data, weights, None, output)
        sys.stdout.write("Processing %d of %d [%.2f%%]   \r" % (i * skip,
                                                                totalsamp / nsamp,
                                                                (i / (float(totalsamp / nsamp) / skip) * 100)))
        sys.stdout.flush()

# Save file
np.save("casa_raw_processed", output)

text_file_name = '/home/lessju/Code/obs' + str(obs) + '_cal' + str(cal) + '.txt' 

text_file = open(text_file_name, 'w')
for i in range(output.shape[1]):
	text_file.write(str(output[0, i]) + '\n')
text_file.close()

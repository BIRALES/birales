import os
import sys
from offline_pointing import Pointing, config
import numpy as np


def beamformer(i, nbeams, data, weights, coeffs, output):
    for b in range(nbeams):
        x = np.dot(data, weights[0, b, :])
        output[b, i] = np.sum(np.power(np.abs(x), 2))


# filepath = "/mnt/2018_02_22/FesCas4/FesCas4_raw.dat"
filepath = "/mnt/2018_03_08/FesCalibTau1/FesCalibTau1_raw.dat"

obs = 'Tau1403'
cal = 'Tau1403'

nsamp = 32768
nants = 32
skip = 4

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
0.732380-0.588059j,
0.668125-0.269481j,
0.816918-0.567270j,
0.684808-0.699143j,
-0.659731-1.178600j,
1.143783+0.104276j,
-0.285521-1.068825j,
0.784679+0.219167j,
0.803180-0.455330j,
0.626265+0.787476j,
-0.252647-0.963801j,
0.823902-0.539061j,
0.914681-0.152378j,
0.314542-0.952656j,
0.555859-0.237510j,
0.495664+0.931462j,
1.043619+0.355521j,
0.696186+0.834885j,
0.975509-0.303480j,
0.638386+0.563067j,
0.086330+1.004608j,
0.991962+0.475933j,
0.877047+0.647834j,
0.855851+0.517210j,
0.510522+1.025221j,
0.952729-0.369845j,
0.966992-0.667751j,
0.235571-1.084553j,
0.779670-0.934907j,
0.947859+0.550121j,
0.157220+0.956486j], dtype=np.complex64)

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
np.save("virgo_raw_processed", output)

#text_file_name = '/home/lessju/Code/obs' + str(obs) + '_cal' + str(cal) + '.txt' 
#
#text_file = open(text_file_name, 'w')
#for i in range(output.shape[1]):
#	text_file.write(str(output[0, i]) + '\n')
#text_file.close()

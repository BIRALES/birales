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
1.303055-0.862737j,
1.461133-0.355642j,
1.607289-0.674936j,
1.187557-1.215521j,
-0.729486-2.091656j,
1.864924+0.494605j,
-0.072313-1.880838j,
1.402667+0.457303j,
1.576394-0.666233j,
0.842352+1.488773j,
-0.059432-1.557383j,
1.320823-0.849953j,
1.551129-0.124948j,
0.799850-1.555624j,
1.137149-0.134736j,
0.740189+1.719167j,
1.700585+0.811918j,
0.854026+1.685043j,
1.584557-0.065660j,
1.076294+1.119282j,
-0.106250+1.707542j,
1.414779+1.068190j,
1.276994+1.480837j,
1.496196+0.892831j,
0.529083+1.733625j,
1.732710-0.295993j,
1.871060-0.781332j,
0.556460-1.636828j,
1.357586-1.297501j,
1.340052+1.150033j,
-0.094246+1.586642j], dtype=np.complex64)

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

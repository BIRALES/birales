from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import time
import sys
import os

filepath = "/mnt/2017/13_04_2017/casa/casa_raw.dat"
nsamp = 262144
totalants = 32
antenna = 0
nants = 32

calibrated_coeffs= np.array([1+0j, 1.12404+21.7j, 1.12408+114.2j, 1.16490+173.9j,
                       1.23148-34.7j, 1.35692+45.7j, 1.30501+41.2j, 1.34344+105.1j,
                       1.01320-73.8j, 1.08150-40.45j, 1.16472+50.2j, 1.17760+43.1j,
                       1.07147-153.4j, 1.10159+64.2j, 1.18613-20.8j, 1.32123+7.4j,
                       1.03883-6.0j, 1.13493-22.5j, 1.11993+51.3j, 1.15337+132.5j,
                       1.05794-104.6j, 1.16382-30.2j, 1.15295+38.5j, 1.22923+90.0j,
                       1.05692-105.8j, 1.09354-118.1j, 1.15251-89.0j, 1.25172+4.5j,
                       1.01880+154.3j, 1.07941-54.4j, 1.08310-121.5j, 1.23430-62.8j],
                            dtype=np.complex64)

new_coeffs = np.array([1.43500487036+5.00183648033j,
                        0.976809788042+14.0120401422j,
                        0.71473392345-16.4677274919j,
                        0.787047654577-12.3541476026j,
                        1.0+0.0j,
                        0.87734341885-13.1926764567j,
                        1.12097855197-10.919491449j,
                        1.12907845757-1.01130729548j,
                        1.35260566748-12.1073711239j,
                        0.934177313648-12.1410981745j,
                        1.03566374156-3.96976136421j,
                        0.949263190405+3.88933843448j,
                        1.07329032978-19.4273723057j,
                        0.830801522624-15.749012328j,
                        1.00922130425-5.83921926162j,
                        0.910517036436-6.0270595963j,
                        1.25787460744-3.74242248587j,
                        1.3023269586-14.3929454746j,
                        1.18488430348-5.46125245053j,
                        0.871927246677+1.23715097281j,
                        1.08103666236-9.4084790126j,
                        1.45375740842-25.7032871463j,
                        1.15232698821-5.72493620401j,
                        1.29137015778-16.8195376425j,
                        1.35508512487+1.07091029569j,
                        1.14912435537-11.5901336617j,
                        1.15790354783-22.3348656395j,
                        1.550546456-38.0468469523j,
                        0.905403999861-21.5057468354j,
                        1.07919607526-15.7264791117j,
                        1.11940577226-7.90189911954j,
                        1.28060038908+7.10802906047j], dtype=np.complex64)

calibrated_coeffs.imag = np.deg2rad(calibrated_coeffs.imag)
new_coeffs.imag = np.deg2rad(new_coeffs.imag)

# Check filesize
filesize = os.path.getsize(filepath)
totalsamp = filesize / (8 * totalants)

# Create output array
output_jack = np.zeros((nants, (totalsamp / nsamp)), dtype=np.float)
output_uncalib = np.zeros((nants, (totalsamp / nsamp)), dtype=np.float)
output_josef = np.zeros((nants, (totalsamp / nsamp)), dtype=np.float)

# Open file
with open(filepath, 'rb') as f:
    for i in range(totalsamp/nsamp):
        data = f.read(nsamp * totalants * 8)
        data = np.frombuffer(data, np.complex64)
        data = data.reshape((1, 1, nsamp, totalants))
        output_jack[:,i] = np.sum(np.abs(data[0,0,:,antenna:antenna+nants])**2, axis = 0)

        # Uncalibrate array
        data = data * (1/np.abs(calibrated_coeffs[antenna:antenna+nants])) * \
                     np.exp(-1j * np.angle(calibrated_coeffs[antenna:antenna+nants]))
        output_uncalib[:,i] = np.sum(np.abs(data[0,0,:,antenna:antenna+nants])**2, axis = 0)

        # Calibrate array
        data = data * new_coeffs[antenna:antenna+nants]
        output_josef[:,i] = np.sum(np.abs(data[0,0,:,antenna:antenna+nants])**2, axis = 0)

        sys.stdout.write("Processing %d of %d [%.2f%%]   \r" % (i,
                                                                totalsamp/nsamp,
                                                                (i / float(totalsamp/nsamp) * 100)))
        sys.stdout.flush()

markers = matplotlib.markers.MarkerStyle.markers.keys()
f = plt.figure()
ax1 = f.add_subplot(111)
ax1.set_title("Jack")
f = plt.figure()
ax2 = f.add_subplot(111)
ax2.set_title("Uncalibrated")
f = plt.figure()
ax3 = f.add_subplot(111)
ax3.set_title("Josef")
for i in range(nants):
    ax1.plot(output_jack[i, :], label="{}".format(i), marker=markers[i])
    ax2.plot(output_uncalib[i, :], label="{}".format(i), marker=markers[i])
    ax3.plot(output_josef[i, :], label="{}".format(i), marker=markers[i])
ax1.legend()
ax2.legend()
ax3.legend()
plt.show()

time.sleep(100000)

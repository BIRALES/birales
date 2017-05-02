from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os

filepath = "/mnt/2017/13_04_2017/casa/casa_raw.dat"
nsamp = 8192
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

updated_coeffs = np.array([1.01025399233-0.525874468674j,
                        0.912474758983-2.46429159751j,
                        0.987591366786+0.878996789064j,
                        0.903964872353-4.23028220055j,
                        1.0+0.0j,
                        0.909813618204-1.91554294227j,
                        0.925284615043-1.84828929251j,
                        0.88245778276-6.96684490118j,
                        1.0205802846+2.04571787155j,
                        0.933549644208+0.366209782714j,
                        0.941040960024+0.417357146742j,
                        0.907460335478-0.215520309997j,
                        0.987831215685-0.404129054545j,
                        0.923084959796-0.778231901328j,
                        0.950912627677-1.13695703956j,
                        0.870831610278-1.70674221298j,
                        0.993661738726-0.754341377051j,
                        0.908181625982-1.32980961291j,
                        0.941244299768-1.71805049958j,
                        0.860387525929-1.96016894699j,
                        0.989941380164-2.14103057855j,
                        0.935918997741-2.86774910111j,
                        0.974010423394-2.9968981212j,
                        0.845659290566-3.92871234745j,
                        1.00408430236-2.74990243283j,
                        0.938693059111+0.122260724303j,
                        0.994065086376-2.79386481128j,
                        0.918698935644-2.85109701518j,
                        1.04423655695-5.86191355696j,
                        0.931038316175-4.68400849313j,
                        0.992008562596-2.21262281278j,
                        0.894036678528-1.60043334635j], dtype=np.complex64)

calibrated_coeffs.imag = np.deg2rad(calibrated_coeffs.imag)
new_coeffs.imag = np.deg2rad(new_coeffs.imag)
updated_coeffs.imag = np.deg2rad(updated_coeffs.imag)

weights = np.full((nants, ), 1+0j, dtype=np.complex64)

# Check filesize
filesize = os.path.getsize(filepath)
totalsamp = filesize / (8 * nants)

# Create output array
output_jack = np.zeros((totalsamp / nsamp), dtype=np.float)
output_josef = np.zeros((totalsamp / nsamp), dtype=np.float)

# Open file
with open(filepath, 'rb') as f:
    for i in range(totalsamp/nsamp):
        data = f.read(nsamp * nants * 8)
        data = np.frombuffer(data, np.complex64)
        data = data.reshape((nsamp, nants))

        # Uncalibrate array
    #    data = data * (1/np.abs(calibrated_coeffs)) * np.exp(-1j * np.angle(calibrated_coeffs))
    #    print('Jack: ', data[0,0,:4,8])

        # Calibrate array
    #    data = data * new_coeffs
    #    print('Josef: ', data[0,0,:4,8])

        output_jack[i] = 10 * np.log10(np.sum(np.abs(np.dot(data, weights))**2))

        data = data * updated_coeffs
        output_josef[i] = 10 * np.log10(np.sum(np.abs(np.dot(data, weights))**2))

        if i == 0:
            # Plotting
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            line1, = ax.plot(output_jack, 'r-', label="Jack")
            line2, = ax.plot(output_josef, 'b-', label="Josef")
            ax.set_ylim([15, 20])
            plt.legend()

        if i % 200 == 0 or i == (totalsamp/nsamp) - 1:
            line1.set_ydata(10*np.log10(output_jack))
            line2.set_ydata(10*np.log10(output_josef))
            plt.pause(0.0001)

        sys.stdout.write("Processing %d of %d [%.2f%%]   \r" % (i,
                                                                totalsamp/nsamp,
                                                                (i / float(totalsamp/nsamp) * 100)))
        sys.stdout.flush()

time.sleep(100000)

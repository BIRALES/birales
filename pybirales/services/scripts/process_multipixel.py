import numpy as np
import struct
from matplotlib import pyplot as plt
from scipy import signal

nbeams = 117
nchans = 8192

if __name__ == "__main__":

    # Open file
    filepath = '/home/lessju/Desktop/Norad41240/2022_08_02/Virgo/Virgo_beam.dat'
    data = np.fromfile(filepath, dtype=float, count=8192*117*4096)
    nsamp = int(data.size / (nchans * nbeams))
    data = np.reshape(data, (nsamp, nchans, nbeams))

    # For each beam
    for i in range(nbeams):

        # Calculate bandpass
        bandpass = np.sum(data[:, :, i], axis=0)

        # Apply median filter bandpass on
        filtered = signal.medfilt(bandpass, 5)
        x = bandpass - filtered
        outliers = np.where(x - np.mean(x) > np.std(x) * 3)
        bandpass = bandpass / nchans
        mean = np.mean(bandpass)

        # Remove outliers
        for item in outliers[0]:
            data[:, item, i] = np.zeros(nsamp) + mean

        # Remove bandpass from data
        data[:, :, i] = data[:, :, i] - bandpass

        # Normalise data
        data[:, :, i] = (data[:, :, i] - np.mean(data[:, :, i])) / np.std(data[:, :, i])

    global_plot = np.zeros((13, 9))

    # Generate multipixel plot
    for i in range(nsamp):

        # Grab array from data
        sample = np.sum(data[i, 2048:8192-2048, :], axis=0) / nchans

        # Reduce output clutter
        sample[np.where(np.abs(sample) < np.std(sample) * 3)] = 0

        for j in range(nbeams):
            if sample[j] != 0:
                global_plot[int(j / 9), j % 9] += 1

    plt.imshow(global_plot.T, aspect='auto', interpolation='nearest')
    plt.xlabel("RA")
    plt.ylabel("DEC")
    plt.colorbar()
    plt.show()

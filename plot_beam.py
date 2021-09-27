from matplotlib import pyplot as plt
import numpy as np

if True:
    filepath = "/storage/data/birales/2021_06_25/Observation_2021-06-25T0810/Observation_2021-06-25T0810_beam.dat"

    d = np.fromfile(filepath, count=8192 * 32 * 4096, dtype=np.float32)
    nsamp = d.shape[0] // (8192 * 32)
    d = np.reshape(d, (nsamp, 8192, 32))

    plt.imshow(d[:, :, 15], aspect='auto')
    plt.colorbar()

    plt.figure()
    plt.plot(np.sum(d[:, :, 15], axis=0))
    plt.show()

else:
    filepath = "/storage/data/birales/2021_06_25/Observation_2021-06-25T0810/Observation_2021-06-25T0810_beam.dat"

    nsamp = 1
    nbeams = 32
    nchans = 8192
    to_plot = np.zeros((1024 * 4 * 8 * 2
        , 3))
    with open(filepath, 'rb') as f:
        counter = 0
        while True:
            d = np.fromfile(f, count=nsamp * nbeams * nchans, dtype=np.float32)

            if d.shape[0] != nsamp * nbeams * nchans:
                break

            d = np.reshape(d, (nsamp, nchans, nbeams))
            d = 10 * np.log10(np.sum(d, axis=(0, 1)) / (nsamp * nchans))
#            for i in range(9):
#                to_plot[counter, i] = d[i + 11]
            to_plot[counter, 0] = d[6]
            to_plot[counter, 1] = d[15]
            to_plot[counter, 2] = d[24]

            counter += 1 
            if counter % 50 == 0:
                print(counter)

    plt.plot(to_plot)
    plt.show()
            
                    

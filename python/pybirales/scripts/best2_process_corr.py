from matplotlib import pyplot as plt
from optparse import OptionParser
import numpy as np
import h5py
import sys

if __name__ == "__main__":

    # Command line options
    p = OptionParser()
    p.set_usage('best2_process_corr.py [options] INPUT_FILE')
    p.set_description(__doc__)

    p.add_option('-b', '--baselines', dest='baseline', action='store', default="0-1",
                 help='Baselines to plot (default: 0-1')
    p.add_option('-a', '--antennas', dest='nants', action='store', default=32,
                 type='int', help='Number of antennas (default: 32)')

    opts, args = p.parse_args(sys.argv[1:])

    if not args:
        print('Please specify an input file! \nExiting.')
        exit(0)

    # Generate baselines
    baselines, counter = {}, 0
    for i in range(opts.nants):
        for j in range(i + 1, opts.nants):
            baselines['{}-{}'.format(i, j)] = counter
            counter += 1

    # Open and read file
    with h5py.File(args[0], "r") as f:
        data = f["Vis"]
        data = data[:]

    # Calculate number of samples in file
    nsamp = len(data)
    try:
        nsamp = np.where(data[:, 0, 0, 0] == 0)[0][0]
    except:
        pass

    # Plot
    for to_plot in opts.baseline.split(','):
        plt.plot(data[:nsamp, 0, baselines[to_plot], 0], label=to_plot)
    plt.legend()
    plt.show()

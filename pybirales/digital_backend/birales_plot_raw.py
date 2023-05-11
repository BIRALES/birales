from pydaq.persisters import RawFormatFileManager
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

from birales_plot import calcAVGSpectra, closest

if __name__ == "__main__":
    from optparse import OptionParser
    from sys import argv, stdout

    parser = OptionParser(usage="usage: %birales_raw_hdf5_plot [options]")
    parser.add_option("--file", action="store", dest="file",
                      type="str", default="", help="Input HDF5 file [default: None]")
    parser.add_option("--resolution", dest="resolution", default=1000, type="int",
                      help="Frequency resolution in KHz (it will be truncated to the closest possible)")
    parser.add_option("--inputlist", dest="inputlist", default="",
                      help="List of TPM inputs to be displayed (default: all)")
    (conf, args) = parser.parse_args(argv[1:])

    DATA_LEN = 1024 * 32
    resolutions = 2 ** np.array(range(16)) * (700000.0 / 2 ** 15)
    rbw = int(closest(resolutions, conf.resolution))
    avg = 2 ** rbw
    nsamples = int(2 ** 15 / avg)
    RBW = (avg * (400000.0 / 16384.0))
    # asse_x = np.arange(nsamples/2 + 1) * RBW * 0.001
    bw = 43750000
    nfreq = int((DATA_LEN / 2 / avg) + 1)
    rbw = bw / (nfreq - 1)
    fc = 25000000
    x1 = fc - (bw / 2.)
    x2 = fc + (bw / 2.)
    asse_x = np.linspace(x1, x2, nfreq)

    if not conf.inputlist == "":
        antenna_list = np.array(conf.inputlist.split(",")) - 1
    else:
        antenna_list = range(32)

    if len(antenna_list) == 1:
        rows = 1
        cols = 1
    elif len(antenna_list) == 2:
        rows = 2
        cols = 1
    elif len(antenna_list) == 32:
        rows = 4
        cols = 8
    else:
        rows = int(np.ceil(np.sqrt(len(antenna_list))))
        cols = int(np.ceil(np.sqrt(len(antenna_list))))

    a = RawFormatFileManager.open_file(conf.file)

    gs = gridspec.GridSpec(rows, cols, wspace=0.6, hspace=1, top=0.9, bottom=0.09, left=0.08, right=0.96)
    fig = plt.figure(figsize=(14, 9), facecolor='w')
    ax = []
    for i in range(len(antenna_list)):
        ax += [fig.add_subplot(gs[i])]
    for k in range(len(antenna_list)):
        for i in a.keys():
            spe, power, rms = calcAVGSpectra(a[i][k], 32)
            ax[k].plot(spe)
        ax[k].set_xlim(3, 510)
        ax[k].set_ylim(-110, 10)
        ax[k].set_title("INPUT-%02d" % (k + 1))
    plt.show()

from pydaq.persisters import RawFormatFileManager
import sys
import datetime
import calendar
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# Antenna mapping
# antenna_mapping = [0, 1, 2, 3, 8, 9, 10, 11, 15, 14, 13, 12, 7, 6, 5, 4]
antenna_mapping = range(16)
nof_samples = 20000000
COLORE = ['b', 'g']


def dt_to_timestamp(d):
    return calendar.timegm(d.timetuple())


def ts_to_datestring(tstamp, formato="%Y-%m-%d %H:%M:%S.%s"):
    return datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(tstamp), formato)


def closest(serie, num):
    return serie.tolist().index(min(serie.tolist(), key=lambda z: abs(z - num)))


def calcSpectra(vett):
    window = np.hanning(len(vett))
    spettro = np.fft.rfft(vett * window)
    N = len(spettro)
    acf = 2  # amplitude correction factor
    cplx = ((acf * spettro) / N)
    spettro[:] = abs((acf * spettro) / N)
    # print len(vett), len(spettro), len(np.real(spettro))
    return np.real(spettro)


def calcolaspettro(dati, nof_samples=32768):
    n = int(nof_samples)  # split and average number, from 128k to 16 of 8k # aavs1 federico
    sp = [dati[x:x + n] for x in range(0, len(dati), n)]
    mediato = np.zeros(len(calcSpectra(sp[0])))
    for k in sp:
        singolo = calcSpectra(k)
        mediato[:] += singolo
    mediato[:] /= (2 ** 15 / nof_samples)  # federico
    with np.errstate(divide='ignore', invalid='ignore'):
        mediato[:] = 20 * np.log10(mediato / 127.0)
    d = np.array(dati, dtype=np.float64)
    adu_rms = np.sqrt(np.mean(np.power(d, 2), 0))
    volt_rms = adu_rms * (1.7 / 256.)
    with np.errstate(divide='ignore', invalid='ignore'):
        power_adc = 10 * np.log10(np.power(volt_rms, 2) / 400.) + 30
    power_rf = power_adc + 12
    return mediato, power_rf


def dB2Linear(valueIndB):
    """
    Convert input from dB to linear scale.
    Parameters
    ----------
    valueIndB : float | np.ndarray
        Value in dB
    Returns
    -------
    valueInLinear : float | np.ndarray
        Value in Linear scale.
    Examples
    --------
    #>>> dB2Linear(30)
    1000.0
    """
    return pow(10, valueIndB / 10.0)


def linear2dB(valueInLinear):
    """
    Convert input from linear to dB scale.
    Parameters
    ----------
    valueInLinear : float | np.ndarray
        Value in Linear scale.
    Returns
    -------
    valueIndB : float | np.ndarray
        Value in dB scale.
    Examples
    --------
    #>>> linear2dB(1000)
    30.0
    """
    return 10.0 * np.log10(valueInLinear)


def dBm2Linear(valueIndBm):
    """
    Convert input from dBm to linear scale.
    Parameters
    ----------
    valueIndBm : float | np.ndarray
        Value in dBm.
    Returns
    -------
    valueInLinear : float | np.ndarray
        Value in linear scale.
    Examples
    --------
    #>>> dBm2Linear(60)
    1000.0
    """
    return dB2Linear(valueIndBm) / 1000.


def linear2dBm(valueInLinear):
    """
    Convert input from linear to dBm scale.
    Parameters
    ----------
    valueInLinear : float | np.ndarray
        Value in Linear scale
    Returns
    -------
    valueIndBm : float | np.ndarray
        Value in dBm.
    Examples
    --------
    #>>> linear2dBm(1000)
    60.0
    """
    return linear2dB(valueInLinear * 1000.)


if __name__ == "__main__":
    from optparse import OptionParser
    from sys import argv, stdout

    parser = OptionParser(usage="usage: %birales_plot_hdf5_raw.py [options]")
    parser.add_option("--file", action="store", dest="file", type=str,
                      default="", help="File name")
    parser.add_option("--skip", action="store", dest="skip", type=int,
                      default=-1, help="Skip N blocks")
    parser.add_option("--start", action="store", dest="start",
                      default="", help="Start time for filter (YYYY-mm-DD_HH:MM:SS)")
    parser.add_option("--stop", action="store", dest="stop",
                      default="", help="Stop time for filter (YYYY-mm-DD_HH:MM:SS)")
    parser.add_option("--startfreq", action="store", dest="startfreq",
                      default=390, help="Plot Start Frequency")
    parser.add_option("--stopfreq", action="store", dest="stopfreq",
                      default=430, help="Plot Stop Frequency")
    parser.add_option("--inputlist", action="store", dest="inputlist",
                      default="", help="List of antenna to plot")
    parser.add_option("--resolution", dest="resolution", default=1000, type="int",
                      help="Frequency resolution in KHz (it will be truncated to the closest possible)")
    parser.add_option("--power", dest="power", default="",
                      help="Compute and Plot Total Power of the given frequency")
    parser.add_option("--spectrogram", action="store_true", dest="spectrogram",
                      default=False, help="Plot Spectrograms")
    parser.add_option("--average", action="store_true", dest="average",
                      default=False, help="Compute the averaged Spectrum")
    parser.add_option("--yticks", action="store_true", dest="yticks",
                      default=False, help="Maximize Y Ticks in Spectrograms")
    parser.add_option("--pol", action="store", dest="pol",
                      default="x", help="Spectrograms Polarization")
    parser.add_option("--lofreq", action="store", dest="lofreq",
                      default=382.23e3, help="IF Local Oscillator Frequency (def: 382.23MHz)")
    parser.add_option("--ddcfreq", action="store", dest="ddcfreq",
                      default=25.77e3, help="ADC/DDC Central Frequency (def: 25.77MHz)")

    (opts, args) = parser.parse_args(argv[1:])

    t_date = None
    t_start = None
    t_stop = None
    t_cnt = 0

    resolutions = 2 ** np.array(range(16)) * 700000.0 / 8. / 2. / 2 ** 15
    rbw = int(closest(resolutions, opts.resolution))
    avg = 2 ** rbw
    nof_samples = int(2 ** 15 / avg)
    RBW = (avg * (350000.0 / 8 / 16384.0))
    asse_x = (np.arange(nof_samples / 2 + 1) * RBW + opts.ddcfreq + opts.lofreq) * 0.001 - (700.0 / 32.)
    # remap = [0, 1, 2, 3, 8, 9, 10, 11, 15, 14, 13, 12, 7, 6, 5, 4]
    # remap = [0, 1, 2, 3, 12, 13, 14, 15, 7, 6, 5, 4, 11, 10, 9, 8]
    remap = range(32)
    # asse_x = range(32768)
    # nof_samples = 32768
    xmin = closest(asse_x, int(opts.startfreq))
    xmax = closest(asse_x, int(opts.stopfreq))
    # xmax = 32768
    # xmin = 0
    if opts.inputlist == "":
        ant_list = np.arange(1, 33)
    else:
        ant_list = opts.inputlist.split(",")
    # print("Frequency resolution set %3.1f KHz" % resolutions[rbw])

    # if opts.date:
    #     try:
    #         t_date = datetime.datetime.strptime(opts.date, "%Y-%m-%d")
    #         t_start = dt_to_timestamp(t_date)
    #         t_stop = dt_to_timestamp(t_date) + (60 * 60 * 24)
    #     except:
    #         print("Bad date format detected (must be YYYY-MM-DD)")
    # else:
    if opts.start:
        try:
            t_start = dt_to_timestamp(datetime.datetime.strptime(opts.start, "%Y-%m-%d_%H:%M:%S"))
            print("Start Time:  " + ts_to_datestring(t_start))
        except:
            print("Bad t_start time format detected (must be YYYY-MM-DD_HH:MM:SS)")
    if opts.stop:
        try:
            t_stop = dt_to_timestamp(datetime.datetime.strptime(opts.stop, "%Y-%m-%d_%H:%M:%S"))
            print("Stop  Time:  " + ts_to_datestring(t_stop))
        except:
            print("Bad t_stop time format detected (must be YYYY-MM-DD_HH:MM:SS)")

    birales_data = RawFormatFileManager.open_file(opts.file)

    if len(ant_list) == 1:
        rows = 1
        cols = 1
    elif len(ant_list) == 2:
        rows = 2
        cols = 1
    elif len(ant_list) == 32:
        rows = 4
        cols = 8
    else:
        rows = int(np.ceil(np.sqrt(len(ant_list))))
        cols = int(np.ceil(np.sqrt(len(ant_list))))
    gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.6, top=0.9, bottom=0.09, left=0.08, right=0.96)
    fig = plt.figure(figsize=(14, 9), facecolor='w')

    meas = {}
    if opts.spectrogram:
        p = 0
        if opts.pol.lower() == "y":
            p = 1
        band = str(opts.startfreq) + "-" + str(opts.stopfreq)
        ystep = 1
        BW = int(band.split("-")[1]) - int(band.split("-")[0])
        if rows == 3:
            ystep = 5
        elif rows == 4:
            ystep = 10
        # ystep = 1
        # xmin = closest(asse_x, int(opts.startfreq))
        # xmax = closest(asse_x, int(opts.stopfreq))

        wclim = (-50, 0)
        # fig.suptitle("Tile-%02d Spectrograms (RBW: %3.1f KHz)" % (int(opts.tile), RBW), fontsize=16)
        allspgram = []
        ax = []
        for num, tpm_input in enumerate(ant_list):
            ant = int(tpm_input) - 1
            ax += [fig.add_subplot(gs[num])]
            allspgram += [[]]
            allspgram[num] = np.empty((10, xmax - xmin + 1,))
            # allspgram[num] = np.empty((10, int(nof_samples/2+1),))
            allspgram[num][:] = np.nan

            ax[num].cla()
            ax[num].set_title("Input-%02d" % int(tpm_input), fontsize=12)
            ax[num].imshow(allspgram[num], interpolation='none', aspect='auto', extent=[xmin, xmax, 60, 0], cmap='jet',
                           clim=wclim)
            # ax[num].imshow(allspgram[num], interpolation='none', aspect='auto', extent=[0, 8192, 60, 0], cmap='jet', clim=wclim)
            ax[num].set_ylabel("MHz")
            ax[num].set_xlabel('time')

    else:
        if opts.power == "":
            # fig.suptitle("Tile-%02d Spectra (RBW: %3.1f KHz)" % (int(opts.tile), RBW), fontsize=16)
            ax = []
            for num, tpm_input in enumerate(ant_list):
                ant = int(tpm_input) - 1
                ax += [fig.add_subplot(gs[num])]
                ax[num].set_title("Input-%02d" % int(tpm_input), fontsize=12)
                # ax[num].set_xlim(asse_x[closest(asse_x, float(opts.startfreq))],
                #                 asse_x[closest(asse_x, float(opts.stopfreq))])
                ax[num].set_ylim(-100, 20)
                ax[num].set_ylabel("dB", fontsize=10)
                ax[num].set_xlabel("MHz", fontsize=10)
                ax[num].tick_params(axis='both', which='major', labelsize=8)
                ax[num].grid()

        else:
            # fig.suptitle("Tile-%02d Power of Frequency Channel %3.1f MHz (RBW: %3.1f KHz)  -  from %s  to  %s" %
            #              (int(opts.tile), float(opts.power), RBW,
            #               ts_to_datestring(fname_to_tstamp(lista[0][-21:-7]), formato="%Y-%m-%d %H:%M:%S"),
            #               ts_to_datestring(fname_to_tstamp(lista[-1][-21:-7]), formato="%Y-%m-%d %H:%M:%S")),
            #              fontsize=14)
            ax = []
            for num, tpm_input in enumerate(ant_list):
                ant = int(tpm_input) - 1
                ax += [fig.add_subplot(gs[num])]
                ax[num].set_title("Input-%02d" % int(tpm_input), fontsize=12)
                ax[num].set_ylim(-20, 20)
                # ax[num].set_xlim(0, len(lista))
                ax[num].set_ylabel("dB", fontsize=10)
                ax[num].set_xlabel("timestamp", fontsize=10)
                ax[num].tick_params(axis='both', which='major', labelsize=8)
                ax[num].grid()
            norm_factor = []

    timestamps = list(birales_data.keys())
    print("\nLoaded %d timestamps\n" % len(timestamps))
    seq = 0
    for t_stamp in timestamps:
        dtimestamp = ts_to_datestring(int(t_stamp), formato="%Y-%m-%d %H:%M:%S")
        seq = seq + 1
        for num, tpm_input in enumerate(ant_list):
            if opts.spectrogram:
                ant = int(tpm_input) - 1
                sys.stdout.write("\r[%d/%d] %s Processing Input-%02d" % (seq, len(timestamps), dtimestamp, ant))
                sys.stdout.flush()
                spettro, rms = calcolaspettro(birales_data[t_stamp][remap[ant]], nof_samples)
                # allspgram[num] = np.concatenate((allspgram[num], [spettro[xmin:xmax + 1]]), axis=0)
                allspgram[num] = np.concatenate((allspgram[num], [spettro[xmin:xmax + 1]]), axis=0)
            elif opts.power == "":
                ant = int(tpm_input) - 1
                # ("%s Processing Input-%02d" % (dtimestamp, ant))
                print("Processing Input-%02d" % (ant))
                for npol, pol in enumerate(["Pol-X", "Pol-Y"]):
                    if opts.average:
                        spettro, rms = calcolaspettro(data[remap[ant], npol, :], nof_samples)
                        if not nn:
                            meas["Input-%02d_%s" % (ant, pol)] = dB2Linear(spettro)
                        else:
                            meas["Input-%02d_%s" % (ant, pol)][:] += dB2Linear(spettro)
                    else:
                        meas["Input-%02d_%s" % (ant, pol)], rms = calcolaspettro(data[remap[ant], npol, :], nof_samples)
                        ax[num].plot(asse_x[3:], meas["Input-%02d_%s" % (ant, pol)][3:], color=COLORE[npol])
                        if (nn == (len(lista) - 1)):
                            if cols == 1:
                                if rows == 1:
                                    ax[num].annotate("RF Power: %3.1f dBm" % rms, (300, 15 - (npol * 5)), fontsize=16,
                                                     color=COLORE[npol])
                                else:
                                    ax[num].annotate("RF Power: %3.1f dBm" % rms, (300, 10 - (npol * 10)), fontsize=16,
                                                     color=COLORE[npol])
                            else:
                                ax[num].annotate("RF Power: %3.1f dBm" % rms, (180 - cols * 7, 5 - (npol * 15)),
                                                 fontsize=14 - cols, color=COLORE[npol])
        # for num, tpm_input in enumerate(opts.inputlist.split(",")):
        #     ant = int(tpm_input) - 1
        #     print("Processing Input-%02d" % (ant))
        #     for npol, pol in enumerate(["Pol-X", "Pol-Y"]):
        #         spettro, rms = calcolaspettro(data[remap[ant], npol, :], nof_samples)
        #         if not nn:
        #             norm_factor += [spettro[closest(asse_x, float(opts.power))]]
        #         meas["Input-%02d_%s" % (ant, pol)] += [spettro[closest(asse_x, float(opts.power))] - norm_factor[num * 2 + npol]]

    if opts.spectrogram:
        for num, tpm_input in enumerate(ant_list):
            first_empty, allspgram[num] = allspgram[num][:10], allspgram[num][10:]
            ax[num].imshow(np.rot90(allspgram[num]), interpolation='none', aspect='auto', cmap='jet', clim=wclim)
            BW = int(band.split("-")[1]) - int(band.split("-")[0])
            ytic = np.array(range(int(BW / ystep) + 1)) * ystep * (len(np.rot90(allspgram[num])) / float(BW))
            ax[num].set_yticks(len(np.rot90(allspgram[num])) - ytic)
            ylabmax = (np.array(range(int(BW / ystep) + 1)) * ystep) + int(band.split("-")[0])
            ax[num].set_yticklabels(ylabmax.astype("str").tolist())
    RawFormatFileManager.close_file(birales_data)
    plt.show()

    # if not opts.power == "":
    #     for num, tpm_input in enumerate(opts.inputlist.split(",")):
    #         ant = int(tpm_input) - 1
    #         for npol, pol in enumerate(["Pol-X", "Pol-Y"]):
    #             ax[num].plot(meas["Input-%02d_%s" % (ant, pol)], color=COLORE[npol])
    #
    # if opts.spectrogram:
    #     for num, tpm_input in enumerate(opts.inputlist.split(",")):
    #         first_empty, allspgram[num] = allspgram[num][:10], allspgram[num][10:]
    #         ax[num].imshow(np.rot90(allspgram[num]), interpolation='none', aspect='auto', cmap='jet', clim=wclim)
    #         BW = int(band.split("-")[1]) - int(band.split("-")[0])
    #         ytic = np.array(range(int(BW / ystep) + 1)) * ystep * (len(np.rot90(allspgram[num])) / float(BW))
    #         ax[num].set_yticks(len(np.rot90(allspgram[num])) - ytic)
    #         ylabmax = (np.array(range(int(BW / ystep) + 1)) * ystep) + int(band.split("-")[0])
    #         ax[num].set_yticklabels(ylabmax.astype("str").tolist())
    #
    # if opts.average:
    #     for num, tpm_input in enumerate(opts.inputlist.split(",")):
    #         ant = int(tpm_input) - 1
    #         print("Processing Input-%02d" % (ant))
    #         for npol, pol in enumerate(["Pol-X", "Pol-Y"]):
    #             meas["Input-%02d_%s" % (ant, pol)] /= len(lista)
    #             ax[num].plot(asse_x[3:], linear2dB(meas["Input-%02d_%s" % (ant, pol)])[3:], color=COLORE[npol])
    #
    #
    #
    #

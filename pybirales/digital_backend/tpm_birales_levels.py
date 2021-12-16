#!/usr/bin/env python

'''
	Equalize levels of TPM inputs to given dBm level
'''

__author__ = "Andrea Mattana"
__copyright__ = "Copyright 2021, Istituto di RadioAstronomia, INAF, Italy"
__credits__ = ["Andrea Mattana"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Andrea Mattana"

import matplotlib
matplotlib.use("tkagg")
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from pybirales.digital_backend.tile_debris import Tile
import numpy as np
from optparse import OptionParser
import datetime
import os
import sys
import time
import socket
import glob
import struct
import warnings
warnings.filterwarnings("ignore")

log_path = "/var/log/frb/DSA/"
DEVNULL = open(os.devnull, 'w')

RXPKT_HEAD = 1
# slave  = 5
RXPKT_MASTER = 124
# RXPKT_CMD    = 99 # ask version
# RXPKT_CMD    = 108 # get port
# RXPKT_CMD = 110 # get_data
RXPKT_CMD = 111  # set_data
RXPKT_COUNT = 1
# RXPKT_DATA_TYPE = 12 # U16
RXPKT_DATA_TYPE = 8  # U8
RXPKT_PORT_TYPE = 4  # DIO
# RXPKT_PORT_NUMBER = 108 # 00_15
# RXPKT_PORT_NUMBER = 96 # 00_07
RXPKT_PORT_NUMBER = 97  # 08_15 # Attenuation


def bit2att(val):
    attenuazione = ((val & 2 ** 0) >> 0) * 0.5
    attenuazione += ((val & 2 ** 1) >> 1) * 16
    attenuazione += ((val & 2 ** 2) >> 2) * 1
    attenuazione += ((val & 2 ** 3) >> 3) * 8
    attenuazione += ((val & 2 ** 4) >> 4) * 2
    attenuazione += ((val & 2 ** 5) >> 5) * 4
    return attenuazione


def att2bit(val):
    # print("\natt2bit(",val,")\tBin:",bin(val),
    val = int(val * 2)
    # print val,")\tBin:",bin(val),
    attenuazione = ((val & 2 ** 0) >> 0) * 1
    attenuazione += ((val & 2 ** 1) >> 1) * 4
    attenuazione += ((val & 2 ** 2) >> 2) * 16
    attenuazione += ((val & 2 ** 3) >> 3) * 32
    attenuazione += ((val & 2 ** 4) >> 4) * 8
    attenuazione += ((val & 2 ** 5) >> 5) * 2
    # print attenuazione,bin(attenuazione)
    return attenuazione


def get_att_value(s, slave):
    RXPKT_CMD = 110  # get_data
    RXPKT_PORT_NUMBER = 97  # 08_15 # Attenuation
    msg = struct.pack('>BBBBBBBBB', RXPKT_HEAD, slave, RXPKT_MASTER, RXPKT_CMD, RXPKT_COUNT, 3, RXPKT_DATA_TYPE,
                      RXPKT_PORT_TYPE, RXPKT_PORT_NUMBER)
    s.send(msg)
    a = s.recv(32)
    if struct.unpack('>' + str(len(a)) + 'B', a)[5] == 0:
        att = bit2att(struct.unpack('>' + str(len(a)) + 'B', a)[10])
    else:
        att = -1
    return att


def set_att_value(s, slave, value):
    value = round(value * 2) / 2  # 0.5 dB is the step for attenuation
    RXPKT_CMD = 111  # set_data
    RXPKT_PORT_NUMBER = 97  # 08_15 # Attenuation
    msg = struct.pack('>BBBBBBBBBB', RXPKT_HEAD, slave, RXPKT_MASTER, RXPKT_CMD, RXPKT_COUNT, 4, RXPKT_DATA_TYPE,
                      RXPKT_PORT_TYPE, RXPKT_PORT_NUMBER, att2bit(value))
    s.send(msg)
    a = s.recv(32)
    if struct.unpack('>' + str(len(a)) + 'B', a)[5] != 0:
        print("Cmd returned an error!!!")


def get_vr_value(s, slave):
    RXPKT_CMD = 110  # get_data
    RXPKT_PORT_NUMBER = 96  # 00_07
    msg = struct.pack('>BBBBBBBBB', RXPKT_HEAD, slave, RXPKT_MASTER, RXPKT_CMD, RXPKT_COUNT, 3, RXPKT_DATA_TYPE,
                      RXPKT_PORT_TYPE, RXPKT_PORT_NUMBER)
    s.send(msg)
    a = s.recv(32)
    if struct.unpack('>' + str(len(a)) + 'B', a)[5] == 0:
        # print("\n\n\n",struct.unpack('>'+str(len(a))+'B',a),"\n\n\n"
        val = struct.unpack('>' + str(len(a)) + 'B', a)[10]
    else:
        val = -1
    return val


def set_vr_value(s, slave, value):
    RXPKT_CMD = 111  # set_data
    RXPKT_PORT_NUMBER = 96  # 00_07
    # print("Setting val:",value
    msg = struct.pack('>BBBBBBBBBB', RXPKT_HEAD, slave, RXPKT_MASTER, RXPKT_CMD, RXPKT_COUNT, 4, RXPKT_DATA_TYPE,
                      RXPKT_PORT_TYPE, RXPKT_PORT_NUMBER, value)
    s.send(msg)
    a = s.recv(32)
    if struct.unpack('>' + str(len(a)) + 'B', a)[5] != 0:
        print("Cmd returned an error!!!")


def stampa_conf(ip, rx_id, att_val, vrval):
    print("\n\nBox IP: %s" % (ip),)
    for i in rx_id:
        print("\tRx-%d" % (i),)
    print("")
    print("-----------------------",)
    for i in rx_id:
        print("--------",)
    print("")

    print("DSA dB Val:\t",)
    for i in att_val:
        print("\t%3.1f" % (i),)
    print("")

    print("IF AMP 1:\t",)
    for i in vrval:
        if ((i & 4) == 4):
            print("\tON",)
        else:
            print("\tOFF",)
    print("")

    print("IF AMP 2:\t",)
    for i in vrval:
        if ((i & 1) == 1):
            print("\tON",)
        else:
            print("\tOFF",)
    print("")

    print("IF AMP 3:\t",)
    for i in vrval:
        if ((i & 2) == 2):
            print("\tON",)
        else:
            print("\tOFF",)
    print("")

    print("IF AMP 4:\t",)
    for i in vrval:
        if ((i & 8) == 8):
            print("\tON",)
        else:
            print("\tOFF",)
    print("")

    print("RF AMP:\t\t",)
    for i in vrval:
        if ((i & 16) == 16):
            print("\tON",)
        else:
            print("\tOFF",)
    print("")

    print("OL AMP:\t\t",)
    for i in vrval:
        if ((i & 128) == 128):
            print("\tON",)
        else:
            print("\tOFF",)
    print("")

    print("DSA Regulator:\t",)
    for i in vrval:
        if ((i & 32) == 32):
            print("\tON",)
        else:
            print("\tOFF",)
    print("")


def open_rx_connections(rx_ip_list):
    s = []
    for i in range(len(rx_ip_list)):
        s += [socket.socket(socket.AF_INET, socket.SOCK_STREAM)]
        s[i].settimeout(1)
        s[i].connect((rx_ip_list[i], 5002))
    return s


def close_rx_connections(s):
    for i in range(len(s)):
        s[i].close()


def get_levels(tile):
    rms = tile.get_adc_rms()
    rfpower = []
    for rms_val in rms:
        with np.errstate(divide='ignore'):
            rfpower += [10 * np.log10(np.power((rms_val * (1.7 / 16384.)), 2) / 400.) + 30 + 12]
    return rms, rfpower


def print_levels(ants, rfpow, att, rms, hide_levels=False, hide_dsa=False):
    ora = datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(time.time()), "%Y-%m-%d %H:%M:%S")
    print("\nMeasurement: " + ora + "\n")
    if not hide_levels and not hide_dsa:
        print("   RxBoxIP   \tCarrier   DSA\tANTENNA\t input\t RMS\t  LEVEL")
        print("\t\t  id       dB\t BEST\t   #\t\t   dBm ")
        print("---------------------------------------------------------------------")
        for k, an in enumerate(ants):
            if not an[0] == "unused":
                print(" %s\t   %s\t  %s\t%s\t   %02d\t%s\t%s" % (rx_ip_list[an[1]], an[2], att[k], an[0], k,
                                                             ("%3.1f" % rms[k]).rjust(6), ("%3.1f" % rfpow[k]).rjust(6)))
    elif hide_dsa:
        print("NTENNA  \t LEVEL")
        print("   BEST\t\t   dBm ")
        print("-------------------------")
        for k, an in enumerate(ants):
            if not an[0] == "unused":
                print(" %s\t\t%s" % (an[0], ("%3.1f" % rfpow[k]).rjust(6)))
    elif hide_levels:
        print(" ANTENNA  DSA")
        print("   BEST\t    dB")
        print("------------------------")
        for k, an in enumerate(ants):
            if not an[0] == "unused":
                print(" %s\t%s" % (an[0], ("% 2.1f" % att[k]).rjust(6)))
    else:
        print(" ANTENNA \t RMS")
        print("   BEST\t  ")
        print("-----------------------")
        for k, an in enumerate(ants):
            if not an[0] == "unused":
                print(" %s\t   %3.1f" % (an[0], rms[k]))
    print()


def read_dsa(netlist, ants, dsa=None, dontsave=False):
    if dsa is None:
        dsa = ([33] * 32)
    rx_att_old = (np.zeros(32) + 33).tolist()
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    else:
        flist = sorted(glob.glob(log_path + "RX-DSA_*.txt"))
        if len(flist):
            with open(flist[-1]) as f:
                data = f.readlines()
            for i in range(len(ants)):
                rx_att_old[i] = float(data[i].split("\t")[1])
    try:
    #if True:
        s = []
        #print netlist

        for i in range(len(netlist)):
            #print("Connecting to " , netlist[i])
            s += [socket.socket(socket.AF_INET, socket.SOCK_STREAM)]
            s[i].connect((netlist[i], 5002))

        rx_atts = []
        for i in range(len(ants)):
            #print("Antenna ", ants[i][0], ants[i][1], ants[i][2])
            if dsa[i] == 33:
                rx_atts += [get_att_value(s[ants[i][1]], ants[i][2])]
            else:
                rx_atts += [dsa[i]]
                if not ants[i][0] == "unused":
                    set_att_value(s[ants[i][1]], ants[i][2], dsa[i])
        for i in range(len(netlist)):
            s[i].close()
        del s

        changed = False
        for n, d in enumerate(rx_atts):
            if not d == rx_att_old[n]:
                changed = True
        if changed and not dontsave:
            with open(log_path + datetime.datetime.strftime(datetime.datetime.utcnow(),
                                                          "RX-DSA_%Y-%m-%d_%H%M%S.txt"), "w") as fdsa:
                for n, d in enumerate(rx_atts):
                    fdsa.write("%s\t%.1f\n" % (ants[n][0], d))
                    fdsa.flush()
    #else:
    except:
        print("Failed reading attenuation...")
        rx_atts = (np.zeros(32) + 33).tolist()
        pass

    return rx_atts


def eq_signals(rx_ip_list, best_rx, eqvalue=None, eqiter=3, verbose=False):
    try:
        for z in range(eqiter):
            sys.stdout.write("\r[%d/%d] Equalization Cycle..." % ((z + 1), eqiter))
            sys.stdout.flush()
            att = read_dsa(rx_ip_list, best_rx, dontsave=True)
            time.sleep(0.3)
            rms, rfpower = get_levels(tile)
            new_att = []
            for n, k in enumerate(best_rx):
                if not k[0] == "unused":
                    diff = eqvalue - rfpower[n]
                    if diff < 0:
                        new_att += [round(np.clip([att[n] + abs(diff)], 0, 31.5)[0] * 2) / 2]
                    else:
                        new_att += [round(np.clip([att[n] - diff], 0, 31.5)[0] * 2) / 2]
            if z == eqiter - 1:
                att = read_dsa(rx_ip_list, best_rx, new_att, dontsave=False)
            else:
                att = read_dsa(rx_ip_list, best_rx, new_att, dontsave=True)
        if verbose:
            print("\n\nEqualization Completed!\n\n...Reading again...\n\n")
    except:
        print("Failed reading attenuation...")
        att = (np.zeros(32) + 33).tolist()
        pass
    return att


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--ip", action="store", dest="ip",
                      default="10.0.10.6", help="IP [default: 10.0.10.6]")
    parser.add_option("--port", action="store", dest="port",
                      type="int", default="10000", help="Port [default: 10000]")
    parser.add_option("--lmc_ip", action="store", dest="lmc_ip",
                      default="10.0.0.100", help="IP [default: 10.0.0.100]")
    parser.add_option("--lmc_port", action="store", dest="lmc_port",
                      type="int", default="4660", help="Port [default: 4660]")
    parser.add_option("--period", action="store", dest="period",
                      type="float", default="5", help="Acquisition period in seconds [default: 5]")
    parser.add_option("--iter", action="store", dest="iter",
                      type="int", default="3", help="Number of iterations for equalization [default: 3]")
    parser.add_option("--folder", action="store", dest="folder",
                      type="str", default="", help="Destination folder to be added to path /data")
    parser.add_option("--eqvalue", dest="eqvalue", default="",
                      help="Equalization Value (if not provided do not equalize)")
    parser.add_option("--dsa", dest="dsa", default="",
                      help="Set all DSA to a specific value")
    parser.add_option("--dsaload", dest="dsa_load", default="",
                      help="Load and set DSA from file")
    parser.add_option("--plot", dest="plot", default=False, action='store_true',
                      help="Plot levels")
    parser.add_option("--hidedsa", dest="hidedsa", default=False, action='store_true',
                      help="Do not print Rx DSA")
    parser.add_option("--hidelevels", dest="hidelevels", default=False, action='store_true',
                      help="Do not print levels")

    (opts, args) = parser.parse_args()

    eqvalue = ""
    if not opts.eqvalue == "":
        eqvalue = float(opts.eqvalue)

    # Create Tile
    tile = Tile(ip=opts.ip, port=opts.port, lmc_ip=opts.lmc_ip, lmc_port=opts.lmc_port)
    tile.connect()

    time.sleep(1)
    nof_signals = 24

    sock = 0
    rx_ip_list = ["192.168.69.1", "192.168.69.2", "192.168.69.3", "192.168.69.4"]
    best_rx = [["1N-1-1", 0, 1],
               ["1N-1-2", 0, 2],
               ["1N-1-3", 0, 3],
               ["1N-1-4", 0, 4],
               ["1N-2-1", 0, 5],
               ["1N-2-2", 0, 6],
               ["1N-2-3", 0, 7],
               ["1N-2-4", 0, 8],
               ["1N-3-1", 1, 1],
               ["1N-3-2", 1, 2],
               ["1N-3-3", 1, 3],
               ["1N-3-4", 1, 4],
               ["1N-4-1", 1, 5],
               ["1N-4-2", 1, 6],
               ["1N-4-3", 1, 7],
               ["1N-4-4", 1, 8],
               ["1N-5-1", 2, 1],
               ["1N-5-2", 2, 2],
               ["1N-5-3", 2, 3],
               ["1N-5-4", 2, 4],
               ["1N-6-1", 2, 5],
               ["1N-6-2", 2, 6],
               ["1N-6-3", 2, 7],
               ["1N-6-4", 2, 8],
               ["1N-7-1", 3, 1],
               ["1N-7-2", 3, 2],
               ["1N-7-3", 3, 3],
               ["1N-7-4", 3, 4],
               ["1N-8-1", 3, 5],
               ["1N-8-2", 3, 6],
               ["1N-8-3", 3, 7],
               ["1N-8-4", 3, 8]]
    try:
        rms, rfpower = get_levels(tile)
        rx_att = read_dsa(rx_ip_list, best_rx, dontsave=True)
        if opts.plot:
            gs = gridspec.GridSpec(2, 2, wspace=0.13, hspace=0.2, left=0.05, right=0.98, bottom=0.12, top=0.96)
            fig = plt.figure(figsize=(17, 9), facecolor='w')
            plt.ion()
            ax_dbm = fig.add_subplot(gs[1, 0])
            ax_dsa = fig.add_subplot(gs[1, 1])
            antennas = []
            for k in best_rx:
                antennas += [k[0]]
            bars = np.arange(len(antennas))
            ax_levels = []
            for b in bars:
                l, = ax_dbm.plot(b, 0, zorder=3, linestyle='None', marker="s", markersize=10)
                ax_levels += [l]
            bar_dsa = ax_dsa.bar(bars, np.zeros(len(antennas)), 0.8, color='g', zorder=3)
            plt.title("BEST 2N")
            ax_dbm.set_ylabel("dBm")
            ax_dbm.set_yticks(range(-20, 10))
            ax_dbm.set_ylim(-15, 5)
            ax_dbm.set_xticks(range(len(antennas)))
            ax_dbm.set_xticklabels(antennas, fontsize=10, rotation=90)
            ax_dsa.set_xticks(range(len(antennas)))
            ax_dsa.set_xticklabels(antennas, fontsize=10, rotation=90)
            ax_dsa.set_yticks(np.arange(0, 33))
            ax_dsa.set_ylim(0, 32)
            ax_dsa.set_ylabel("dB")
            ax_dsa.set_xlabel("Ricevitori")
            ax_dbm.set_xlim(-1, len(antennas))
            ax_dsa.set_xlim(-1, len(antennas))
            ax_dbm.set_title("Livello segnale misurato dalla TPM")
            ax_dbm.set_title("Attenuazioni impostate nei Ricevitori")
            ax_dbm.grid(zorder=0)
            ax_dsa.grid(zorder=0)
            ax_chart = fig.add_subplot(gs[0, 0:2])
            lines = []
            livelli = []
            for i in range(len(antennas)):
                q = np.empty(200)
                q[:] = np.nan
                livelli += [q.tolist()]
                l, = ax_chart.plot(range(200), livelli[i])
                lines += [l]
            ax_chart.set_xlabel("time samples")
            ax_chart.set_title(str(len(antennas)) + " Segnali della 2-Nord, Total Power Chart")
            ax_chart.set_ylabel("dBm")
            ax_chart.set_ylim(-15, 5)
            ax_chart.set_xlim(0, 200)
            ax_chart.grid(zorder=0)
            fig.canvas.draw()
            plt.show()

            cont = 50
            while not cont == -1:
                rms, rfpower = get_levels(tile)
                print_levels(best_rx, rfpower, rx_att, rms, hide_levels=opts.hidelevels, hide_dsa=opts.hidedsa)

                for i in range(len(antennas)):
                    if opts.plot:
                        livelli[i] = (livelli[i][1:] + [rfpower[i]])
                        lines[i].set_ydata(livelli[i])
                        ax_levels[i].set_ydata(rfpower[i])
                if opts.plot:
                    ax_dbm.set_title("Livello segnale misurato dalla TPM")
                    ax_dbm.set_xlabel(datetime.datetime.strftime(datetime.datetime.utcnow(),
                                                                 "Ultimo aggiornamento %Y-%m-%d %H:%M:%S UTC"))

                if (cont % 50) == 0:
                    rx_att = read_dsa(rx_ip_list, best_rx, dontsave=True)
                    if opts.plot:
                        for i, r in enumerate(rx_att):
                            bar_dsa[i].set_height(r)

                        ax_dsa.set_title("Attenuazioni impostate nei Ricevitori")
                        ax_dsa.set_xlabel("Ultima lettura " + datetime.datetime.strftime(datetime.datetime.utcnow(),
                                                                                 "%Y-%m-%d %H:%M:%S UTC"))

                        fig.canvas.draw()

                #print_levels(antennas, powA, rx_att)
                cont = cont + 1
                if cont > 50:
                    cont = 0
                if not opts.plot:
                    cont = -1
                else:
                    time.sleep(3)

        else:
            sock = 0
            rms, rfpower = get_levels(tile)
            rx_att = read_dsa(rx_ip_list, best_rx, dontsave=True)
            print_levels(best_rx, rfpower, rx_att, rms, hide_levels=opts.hidelevels, hide_dsa=opts.hidedsa)
            if not opts.eqvalue == "":
                rx_att = eq_signals(rx_ip_list, best_rx, eqvalue=float(opts.eqvalue), eqiter=opts.iter, verbose=False)
                time.sleep(0.3)
                rms, rfpower = get_levels(tile)
                print_levels(best_rx, rfpower, rx_att, rms, hide_levels=opts.hidelevels, hide_dsa=opts.hidedsa)
                #print()
            elif not opts.dsa == "":
                dsa = float(opts.dsa)
                if 0 <= dsa < 32:
                    rx_att = read_dsa(rx_ip_list, best_rx, dsa=[dsa]*len(best_rx), dontsave=False)
                    rms, rfpower = get_levels(tile)
                    print_levels(best_rx, rfpower, rx_att, rms, hide_levels=opts.hidelevels, hide_dsa=opts.hidedsa)
                else:
                    print("\n\nERROR: DSA Value must be a value within the range 0-31.5\n")
            elif not opts.dsa_load == "":
                if os.path.exists(opts.dsa_load):
                    print("\nReloading saved DSA values...\n")
                    try:
                        with open(opts.dsa_load) as ffdsa:
                            dati = ffdsa.readlines()
                        data = []
                        for d in dati:
                            data += [float(d.split("\t")[1])]
                        #print "Ricarica: ", data
                        rx_att = read_dsa(rx_ip_list, best_rx, data, dontsave=False)
                        time.sleep(0.3)
                        rms, rfpower = get_levels(tile)
                        print_levels(best_rx, rfpower, rx_att, rms, hide_levels=opts.hidelevels, hide_dsa=opts.hidedsa)
                        print("")
                    except:
                        print("\nMalformed data in the given DSA file (%s)\n" % opts.dsa_load)
                else:
                    print("\nThe given DSA file does not exist (%s)\n" % opts.dsa_load)

    except KeyboardInterrupt:
        print("\nExiting...\n\n")


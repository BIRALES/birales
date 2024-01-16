import time
import sys
import os
import socket
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from struct import *
from optparse import OptionParser
import datetime

DATA_LEN = 32768
CHANNELS = 32
# ASSE_X = np.linspace(0, 400, DATA_LEN/2 + 1)

# adu_remap = [14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1,
#              31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16]
adu_remap = range(32)
lock_file = "/storage/data/tmp/lock"


def fft(data):
    # data = np.frombuffer(data, dtype=np.int8)
    window = np.hanning(len(data))
    output = np.fft.rfft(data * window)
    n = len(output)
    acf = 2  # amplitude correction factor
    output = abs((acf * output) / n)
    # print output
    power = output.max()
    # print power
    return power, np.real(output)


class SpeadRx:
    def __init__(self, write_hdf5):

        self.write_hdf5 = write_hdf5

        self.nof_signals = 32
        self.nof_fpga = 2
        self.nof_signals_per_fpga = self.nof_signals / self.nof_fpga
        self.data_width = 16
        self.data_byte = self.data_width / 8
        self.byte_per_packet = 1024
        self.word_per_packet = self.byte_per_packet / (self.data_width / 8)
        self.fpga_buffer_size = 64 * 1024
        self.nof_samples = self.fpga_buffer_size / 2
        self.expected_nof_packets = self.nof_signals * (self.fpga_buffer_size / self.byte_per_packet) / 2

        self.data_reassembled = np.zeros((self.nof_signals, int(self.fpga_buffer_size / 2)), dtype=np.int16)
        self.line = [0] * self.nof_signals
        self.is_spead = 0
        self.logical_channel_id = 0
        self.packet_counter = 0
        self.payload_length = 0
        self.center_frequency = 0
        # raw_data = 0, raw_synchronized = 1
        # burst_channel = 4, continuous_channel = 5, integrated_channel = 6, multiple_channel = 7
        # tile_beam = 8, integrated_tile_beam = 9, station_beam = 10, integrated_station_beam = 11
        self.capture_mode = 0
        self.timestamp = 0
        self.sync_time = 0
        self.offset = 13 * 8
        self.start_antenna_id = 0
        self.nof_included_antenna = 0

        self.prev_timestamp = 0
        self.recv_packets = 0
        self.first_fpga_id = range(self.nof_fpga)

        if self.write_hdf5:
            _time = time.strftime("%Y%m%d_%H%M%S")
            self.hdf5_channel = h5py.File('channel_data_' + _time + '.hdf5', 'a')

        self.num = 0
        self.plot_init = 0
        self.first_packet = 1

    def spead_header_decode(self, pkt):
        items = unpack('>' + 'Q' * 9, pkt[0:8 * 9])
        self.is_spead = 0
        for idx in range(len(items)):
            spead_item = items[idx]
            spead_id = spead_item >> 48
            val = spead_item & 0x0000FFFFFFFFFFFF
            if spead_id == 0x5304 and idx == 0:
                self.is_spead = 1
            elif spead_id == 0x8001:
                heap_counter = val
                self.packet_counter = heap_counter & 0xFFFFFF
                self.logical_channel_id = heap_counter >> 24
            elif spead_id == 0x8004:
                self.payload_length = val
            elif spead_id == 0x9027:
                self.sync_time = val
            elif spead_id == 0x9600:
                self.timestamp = val
            elif spead_id == 0xA000:
                self.start_antenna_id = (val & 0x000000000000FF00) >> 8
                self.nof_included_antenna = val & 0x00000000000000FF
            elif spead_id == 0xA001:
                self.fpga_id = val & 0x00000000000000FF
            elif spead_id == 0xA002:
                self.start_channel_id = (val & 0x000000FFFF000000) >> 24
                self.start_antenna_id = (val & 0x000000000000FF00) >> 8
            elif spead_id == 0xA003 or spead_id == 0xA001:
                pass
            elif spead_id == 0xA004:
                self.capture_mode = val
            elif spead_id == 0x3300:
                self.offset = 9 * 8
            else:
                print("Error in SPEAD header decoding!")
                print("Unexpected item " + hex(spead_item) + " at position " + str(idx))

    def set_buffers(self):
        self.nof_signals = 32
        self.nof_fpga = 2
        self.nof_signals_per_fpga = self.nof_signals / self.nof_fpga
        self.data_width = 16
        self.data_byte = self.data_width / 8
        self.byte_per_packet = self.payload_length
        self.word_per_packet = self.byte_per_packet / (self.data_width / 8)
        self.nof_samples = self.fpga_buffer_size / 2
        self.expected_nof_packets = self.nof_signals * (self.fpga_buffer_size / self.byte_per_packet)
        self.data_reassembled = np.zeros((self.nof_signals, self.nof_samples), dtype=np.int16)

    def write_buff(self, data):
        idx = (self.packet_counter * self.word_per_packet) % (self.fpga_buffer_size / self.data_byte)
        self.data_reassembled[self.start_antenna_id, idx: idx + self.word_per_packet] = data
        self.recv_packets += 1

    def buffer_demux(self):
        if self.nof_included_antenna == 1:
            self.data_buff = self.data_reassembled
        else:
            print("Synchronised RAW data not supported!")
            exit()

    def detect_full_buffer(self):
        # print self.recv_packets
        if self.packet_counter == 0:
            if self.fpga_id in self.first_fpga_id:
                self.recv_packets = 1
                self.first_fpga_id = [self.fpga_id]
            else:
                self.first_fpga_id = range(self.nof_fpga)
        if self.recv_packets == self.expected_nof_packets:
            self.recv_packets = 0
            return True
        else:
            return False

    def get_raw_data(self, sock):
        while True:
            packet_ok = 0
            try:
                _pkt, _addr = sock.recvfrom(1024 * 10)
                packet_ok = 1
            except socket.timeout:
                sys.stdout.write("\rWaiting for data...                            ")
                sys.stdout.flush()
            except KeyboardInterrupt:
                sys.stdout.write("\nClosed by user\n")
                sys.stdout.flush()
                exit()

            if packet_ok:
                self.spead_header_decode(_pkt)

                if self.is_spead and self.capture_mode == 0:  # is a SPEAD packet and contains raw data
                    if self.first_packet == 1:
                        self.set_buffers()
                        self.first_packet = 0
                    self.write_buff(unpack('<' + 'h' * (self.payload_length / self.data_byte), _pkt[self.offset:]))
                    buffer_ready = self.detect_full_buffer()
                    if buffer_ready:
                        self.buffer_demux()
                        if self.write_hdf5:
                            self.hdf5_channel.create_dataset(str(self.timestamp), data=self.data_buff)
                        self.num += 1
                        tstamp = datetime.datetime.utcnow()
                        sys.stdout.write("\n" + datetime.datetime.strftime(tstamp,
                                                                           "%Y-%m-%d %H:%M:%S ") + "Full buffer received: " + str(
                            self.num) + "    ")
                        sys.stdout.flush()
                        return self.data_buff.tolist()

        self.hdf5_channel.close()

    def plot_raw_data(self, data):
        if self.plot_init == 0:
            plt.ion()
            plt.figure(0)
            plt.title("Raw data")
            self.line[0], = plt.plot([0] * self.nof_samples)
            self.line[0].set_xdata(np.arange(self.nof_samples))
            self.plot_init = 1

        plt.figure(0)
        plt.clf()
        plt.title("RAW data %d" % self.num)
        for n in range(self.nof_signals):
            plt.plot(data[n][0:128])
            plt.draw()
        plt.pause(0.0001)


def dBFS(dati):
    return 10 * np.log10(np.sum(np.abs(np.array(dati) / 128.) ** 2) / len(np.array(dati) / 128.)) - 10 * np.log10(0.5)


def readfile(filename):
    with open(filename, "rb") as f:
        vettore = f.read()
    vett = struct.unpack(str(len(vettore) / 2) + 'h', vettore)
    return vett


def readRAW():
    data = np.zeros((CHANNELS, DATA_LEN), dtype=np.int8)
    for adc_input in range(CHANNELS):
        if os.path.exists("/storage/data/tmp/input_%02d.raw" % adc_input):
            f = open("/storage/data/tmp/input_%02d.raw" % adc_input, "rb")
            dati = f.read()
            f.close()
            buff = struct.unpack(">" + len(dati) * "b", dati)
            data[adc_input] += np.array(buff, dtype=np.int16)
    return data


def calcSpectra(vett):
    window = np.hanning(len(vett))
    spettro = np.fft.rfft(vett * window)
    N = len(spettro)
    acf = 2  # amplitude correction factor
    spettro[:] = abs((acf * spettro) / N)
    # print len(vett), len(spettro), len(np.real(spettro))
    return np.real(spettro)


def calcAVGSpectra(raw_data, avg_num):
    chunk_len = DATA_LEN / avg_num
    spettri = np.zeros((chunk_len / 2) + 1)
    sp = [raw_data[i:i + chunk_len] for i in xrange(0, len(raw_data), chunk_len)]
    for k in sp:
        spettro = calcSpectra(k)
        spettri[:] += np.array(spettro)
    spettri[:] /= avg_num
    with np.errstate(divide='ignore', invalid='ignore'):
        spettri[:] = 20 * np.log10(spettri / ((2 ** 13) - 1))
    adu_rms = np.sqrt(np.mean(np.power(raw_data, 2), 0))
    volt_rms = adu_rms * (1.7 / 16384.)  # VppADC9680/2^bits * ADU_RMS
    with np.errstate(divide='ignore', invalid='ignore'):
        # 10*log10(Vrms^2/Rin) in dBWatt, +3 decadi per dBm
        power_adc = 10 * np.log10(np.power(volt_rms, 2) / 400.) + 30
    power_rf = power_adc + 12  # single ended to diff net loose 12 dBm
    return spettri, power_rf, adu_rms
    # return spettri


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-p",
                      dest="port",
                      default="4660",
                      help="UDP port")
    parser.add_option("-w", "--write_hdf5",
                      dest="write_hdf5",
                      default=False,
                      action="store_true",
                      help="Write HDF5 files")
    parser.add_option("--plot",
                      dest="plot",
                      default=False,
                      action="store_true",
                      help="Generates data plots")
    parser.add_option("--rms",
                      dest="rms",
                      default=False,
                      action="store_true",
                      help="Show RMS")
    parser.add_option("--fft",
                      dest="fft",
                      default=False,
                      action="store_true",
                      help="FFT plots")
    parser.add_option("--raw",
                      dest="raw",
                      default=False,
                      action="store_true",
                      help="RAW plots")
    parser.add_option("--save_raw",
                      dest="save_raw",
                      default=False,
                      action="store_true",
                      help="Save raw (bin) files for RF performance")
    parser.add_option("--channel",
                      dest="channel",
                      default=0,
                      help="ADC input channel to plot [0 (def) to 31]")
    parser.add_option("--avg",
                      dest="avg",
                      default=32, type=int,
                      help="Spectra Average number (def. 32)")
    parser.add_option("--maxhold",
                      dest="maxhold",
                      default=False,
                      action="store_true",
                      help="Enable Max Hold Trace")
    parser.add_option("--minhold",
                      dest="minhold",
                      default=False,
                      action="store_true",
                      help="Enable Min Hold Trace")
    parser.add_option("--ylim",
                      dest="ylim",
                      default="-90,-10",
                      help="Y-Range on fft Plot, def: -90,-10")
    parser.add_option("--text",
                      dest="text",
                      default=False,
                      action="store_true",
                      help="Print Fundamental Tone info under plot")
    parser.add_option("--plot_title",
                      dest="plot_title",
                      default="",
                      help="FFT Plot title")

    (options, args) = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
    sock.bind(("0.0.0.0", 4660))
    sock.settimeout(1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)

    spead_rx_inst = SpeadRx(options.write_hdf5)

    # Removing Lock File
    if os.path.exists(lock_file):
        os.system("rm -rf " + lock_file)

    spettro = []

    canale = "CH-%02d" % int(options.channel)

    if options.plot and options.fft:
        plt.ion()
        if options.text:
            gs = gridspec.GridSpec(2, 1, height_ratios=[6, 1], hspace=0.2, bottom=0.04, top=0.92)
            fig = plt.figure(figsize=(9, 6), facecolor='w')
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
        else:
            gs = gridspec.GridSpec(1, 1, hspace=0.2, bottom=0.1, top=0.92)
            fig = plt.figure(figsize=(9, 6), facecolor='w')
            ax1 = fig.add_subplot(gs[0])

        ax1.cla()
        nfreq = int((DATA_LEN / 2 / int(options.avg)) + 1)
        bw = 43750000
        rbw = bw / (nfreq - 1)
        # fc = 408447265.625
        fc = 29907226.562500
        # fc = 408000000
        x1 = fc - (bw / 2.)
        x2 = fc + (bw / 2.)
        asse_x = np.linspace(x1, x2, int(nfreq))
        # asse_x = np.arange(nfreq) * (bw / 1. / nfreq) + x1
        line1, = ax1.plot(range(nfreq), color='b', label="Live Spectrum")
        if options.maxhold:
            max_hold = np.zeros(nfreq) - 150
            line2, = ax1.plot(max_hold, color='g', label="Max Hold")
        if options.minhold:
            min_hold = np.zeros(nfreq)
            line3, = ax1.plot(min_hold, color='r', label="Min Hold")
        # ax1.set_yticks()
        ax1.set_xlim(0, nfreq)
        ylim_top = float(options.ylim.split(",")[1])
        ylim_bottom = float(options.ylim.split(",")[0])
        ax1.set_ylim(ylim_bottom, ylim_top)
        title1 = ax1.set_title("Starting Measurements...")
        start_time = ""
        start_time_label = ax1.annotate("Start Time: ", (5, ylim_top - 4), fontsize=12, color='b')
        current_time_label = ax1.annotate("Current Time: ", (5, ylim_top - 9), fontsize=12, color='b')
        rfpower_label = ax1.annotate("RF Power:", (5, ylim_top - 15), fontsize=12, color='g')
        rfrms_label = ax1.annotate("RMS:", (5, ylim_top - 20), fontsize=12, color='g')
        ax1.set_xlabel("MHz", fontsize=14)
        ax1.set_ylabel("dB", fontsize=16)
        ticks = np.arange(9) * (nfreq / 8)
        ax1.set_xticks(ticks)
        # ax1.set_xticklabels(["%3.1f"%(float((asse_x[t]/1000000.))) for t in ticks], fontsize=10)
        ax1.grid()
        ax1.legend()
        if options.text:
            ax2.cla()
            ax2.plot(range(100), color='w')
            ax2.set_axis_off()
            text1 = ax2.annotate("Foundamental Tone Frequency: -", (1, 52), fontsize=10)
            text2 = ax2.annotate("Foundamental Tone Power: -", (1, 7), fontsize=10)
        # fig.tight_layout()
        fig.canvas.draw()
        plt.show()

    while True:
        # os.system("rm -rf /storage/data/tmp/in*")  # data file before
        # os.system("rm -rf /storage/data/tmp/*")  # then lock file

        raw_data = spead_rx_inst.get_raw_data(sock)
        # if not os.path.exists(lock_file):
        current_time = datetime.datetime.utcnow()
        data = datetime.datetime.strftime(current_time, "%Y-%m-%d")
        ora = datetime.datetime.strftime(current_time, "%H%M%S")
        if options.plot:
            if options.fft:
                # media = calcAVGSpectra(path, int(options.channel))
                # spettro = calcSpectra(raw_data[int(options.channel)])
                spettro, power_rms, adu_rms = calcAVGSpectra(np.array(raw_data[int(options.channel)]), options.avg)
                # with np.errstate(divide='ignore', invalid='ignore'):
                #    spettro = 20 * np.log10(spettro / (2 ** (spead_rx_inst.data_width - 1) - 1))

                line1.set_ydata(spettro)
                if options.maxhold:
                    max_hold = np.maximum(max_hold, spettro)
                    line2.set_ydata(max_hold)
                if options.minhold:
                    min_hold = np.minimum(min_hold, spettro)
                    line3.set_ydata(min_hold)
                if not options.plot_title:
                    title1.set_text("Spectrum of ADC Input Channel " + str(options.channel))
                else:
                    title1.set_text(options.plot_title)
                if not start_time:
                    start_time = datetime.datetime.strftime(datetime.datetime.utcnow(), "%Y-%m-%d %H:%M:%S UTC")
                    start_time_label.set_text("Start Time: " + start_time)
                current_time_label.set_text("Curr. Time: " + datetime.datetime.strftime(current_time,
                                                                                        "%Y-%m-%d %H:%M:%S UTC"))
                rfpower_label.set_text("RF Power: %3.1f dBm" % power_rms)
                rfrms_label.set_text("RMS: %d" % int(adu_rms))
                if options.text:
                    tone_chan = np.where(spettro == spettro.max())[0][0]
                    tone_freq = asse_x[tone_chan]
                    # print tone_chan, media[tone_chan], tone_freq
                    text1.set_text("Foundamental Tone Frequency: %3.3f MHz" % (tone_freq / 1000000.))
                    text2.set_text("Foundamental Tone Power: %3.1f dBFS" % spettro[tone_chan])
                fig.canvas.draw()
                # fig.canvas.flush_events()

                sys.stdout.write("\rPlot Spectrum of ADC Input Channel " + str(options.channel) + "        ")
                sys.stdout.flush()

            if options.raw:
                spead_rx_inst.plot_raw_data(raw_data)

        if options.rms:
            for n in range(spead_rx_inst.nof_signals):
                output, ssp = fft(raw_data[n])
                if options.rms:
                    sys.stdout.write("\nPower of Channel %02d:\t%d" % (adu_remap[n], output))
                    sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()

        if options.save_raw:
            for n in range(spead_rx_inst.nof_signals):
                if n == int(options.channel):
                    if not os.path.exists("/storage/data/" + data):
                        os.mkdir("/storage/data/" + data)
                    if not os.path.exists("/storage/data/" + data + "/" + canale):
                        os.mkdir("/storage/data/" + data + "/" + canale)
                    raw_filename = "/storage/data/" + data + "/" + canale + "/" + canale + "_" + data + "_" + ora + ".raw"
                    raw_file = open(raw_filename, "wb")
                    raw_file.write(pack(">" + str(len(raw_data[n])) + "h", *raw_data[n]))
                    raw_file.close()
            # os.system("touch " + lock_file)

        # for n, r in enumerate(raw_data):
        #     if 0 in r:
        #         print "(0) LSB usati in input ", n, r[0:9]
        #     if -1 in r:
        #         print "(-1) LSB usati in input ", n, r[0:9]
        #     if -2 in r:
        #         print "(-2) LSB usati in input ", n, r[0:9]
        #     if 1 in r:
        #         print "(1) LSB usati in input ", n, r[0:9]

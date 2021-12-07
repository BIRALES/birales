#!/usr/bin/env python
from pybirales.digital_backend.tile_debris import Tile
from pyfabil import Device
from pybirales.digital_backend.digital_backend import Station, load_station_configuration, load_configuration_file, apply_config_file
import yaml
import h5py
import datetime
import threading
import logging
import time
import socket
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from struct import unpack
COLOR = ['b', 'g']

DATA_LEN = 32768
# Define default configuration
configuration = {'tiles': None,
                 'station': {
                     'id': 0,
                     'name': "debris",
                     "number_of_antennas": 32,
                     'program': False,
                     'initialise': False,
                     'program_cpld': False,
                     'enable_test': False,
                     'bitfile': None,
                     'channel_truncation': 7,
                     'channel_integration_time': -1,
                     'ada_gain': None
                     },
                 'observation': {
                     'sampling_frequency': 700e6,
                     'bandwidth': 12.5e6,
                     'ddc_frequency': 139.65e6
                     },
                 'network': {
                     'lmc': {
                         'tpm_cpld_port': 10000,
                         'lmc_ip': "10.0.10.201",
                         'use_teng': True,
                         'lmc_port': 4660,
                         'lmc_mac': 0x248A078F9D38,
                         'integrated_data_ip': "10.0.0.2",
                         'integrated_data_port': 5000,
                         'use_teng_integrated': True},
                     'csp_ingest': {
                         'src_ip': "10.0.10.254",
                         'dst_mac': 0x248A078F9D38,
                         'src_port': None,
                         'dst_port': 4660,
                         'dst_ip': "10.0.10.200",
                         'src_mac': None}
                    }
                 }


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


def calcAVGSpectra(raw_data, avg_num):
    chunk_len = int(DATA_LEN / avg_num)
    spettri = np.zeros(int((chunk_len / 2)) + 1)
    sp = [raw_data[i:i + chunk_len] for i in range(0, len(raw_data), chunk_len)]
    for k in sp:
        spettro = calcSpectra(k)
        spettri[:] += np.array(spettro)
    spettri[:] /= avg_num
    with np.errstate(divide='ignore', invalid='ignore'):
        spettri[:] = 20 * np.log10(spettri / ((2**13) - 1))
    adu_rms = np.sqrt(np.mean(np.power(raw_data, 2), 0))
    volt_rms = adu_rms * (1.7 / 16384.)  # VppADC9680/2^bits * ADU_RMS
    with np.errstate(divide='ignore', invalid='ignore'):
        # 10*log10(Vrms^2/Rin) in dBWatt, +3 decadi per dBm
        power_adc = 10 * np.log10(np.power(volt_rms, 2) / 400.) + 30
    power_rf = power_adc + 12  # single ended to diff net loose 12 dBm
    return spettri, power_rf, adu_rms


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
        self.fpga_buffer_size = 64*1024
        self.nof_samples = self.fpga_buffer_size / 2
        self.expected_nof_packets = self.nof_signals * (self.fpga_buffer_size / self.byte_per_packet) / 2

        self.data_reassembled = np.zeros((self.nof_signals, int(self.fpga_buffer_size/2)), dtype=np.int16)
        self.line = [0]*self.nof_signals
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
            self.hdf5_channel = h5py.File('/storage/data/raw/channel_data_' + _time + '.hdf5', 'a')

        self.num = 0
        self.plot_init = 0
        self.first_packet = 1

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # UDP
        self.sock.bind(("0.0.0.0", 4660))
        self.sock.settimeout(1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2*1024*1024)

    def spead_header_decode(self, pkt):
        items = unpack('>' + 'Q'*9, pkt[0:8*9])
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
                self.offset = 9*8
            else:
                print("Error in SPEAD header decoding!")
                print("Unexpected item " + hex(spead_item) + " at position " + str(idx))

    def set_buffers(self):
        self.nof_signals = 32
        self.nof_fpga = 2
        self.nof_signals_per_fpga = int(self.nof_signals / self.nof_fpga)
        self.data_width = 16
        self.data_byte = int(self.data_width / 8)
        self.byte_per_packet = self.payload_length
        self.word_per_packet = int(self.byte_per_packet / (self.data_width / 8))
        self.nof_samples = int(self.fpga_buffer_size / 2)
        self.expected_nof_packets = int(self.nof_signals * (self.fpga_buffer_size / self.byte_per_packet))
        self.data_reassembled = np.zeros((self.nof_signals, self.nof_samples), dtype=np.int16)

    def write_buff(self, pdata):
        idx = int((self.packet_counter * self.word_per_packet) % (self.fpga_buffer_size / self.data_byte))
        self.data_reassembled[self.start_antenna_id, idx: idx + self.word_per_packet] = pdata
        self.recv_packets += 1

    def buffer_demux(self):
        if self.nof_included_antenna == 1:
            self.data_buff = self.data_reassembled
        else:
            print("Synchronised RAW data not supported!")
            exit()

    def detect_full_buffer(self):
        #print self.recv_packets
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

    def get_raw_data(self):
        while True:
            packet_ok = 0
            try:
                _pkt, _addr = self.sock.recvfrom(1024*10)
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
                    self.write_buff(unpack('<' + 'h' * int(self.payload_length / self.data_byte), _pkt[self.offset:]))
                    buffer_ready = self.detect_full_buffer()
                    if buffer_ready:
                        self.buffer_demux()
                        if self.write_hdf5:
                            self.hdf5_channel.create_dataset(str(self.timestamp), data=self.data_buff)
                        self.num += 1
                        tstamp = datetime.datetime.utcnow()
                        #sys.stdout.write("\n" + datetime.datetime.strftime(tstamp, "%Y-%m-%d %H:%M:%S ") + "Full buffer received: " + str(self.num) + "    ")
                        #sys.stdout.flush()
                        return self.data_buff.tolist()

        self.hdf5_channel.close()


    def plot_raw_data(self, data):
        if self.plot_init == 0:
            plt.ion()
            plt.figure(0)
            plt.title("Raw data")
            self.line[0], = plt.plot([0]*self.nof_samples)
            self.line[0].set_xdata(np.arange(self.nof_samples))
            self.plot_init = 1

        plt.figure(0)
        plt.clf()
        plt.title("RAW data %d" % self.num)
        for n in range(self.nof_signals):
            plt.plot(data[n][0:128])
            plt.draw()
        plt.pause(0.0001)


if __name__ == "__main__":
    from optparse import OptionParser
    from sys import argv, stdout
    #global data_received

    parser = OptionParser(usage="usage: %station [options]")
    parser.add_option("--config", action="store", dest="config",
                      type="str", default=None, help="Configuration file [default: None]")
    parser.add_option("--resolution", dest="resolution", default=1000, type="int",
                      help="Frequency resolution in KHz (it will be truncated to the closest possible)")
    parser.add_option("--inputlist", dest="inputlist", default="",
                      help="List of TPM inputs to be displayed (default: all)")
    parser.add_option("--tile", dest="tile", default=1, type=int,
                      help="Tile number to plot")
    parser.add_option("--num", dest="num", default=1000000, type=int,
                      help="Number of acquisitions (def. 1000000)")
    parser.add_option("--nic", dest="nic", default="", type=str,
                      help="If given force to use this NIC")
    parser.add_option("--use_teng", action="store_true", dest="use_teng",
                      default=None, help="Use 10G for LMC (default: None)")
    parser.add_option("--chan-trunc", action="store", dest="chan_trunc",
                      default=None, type="int", help="Channeliser truncation [default: None]")
    parser.add_option("-B", "--beamf_start", action="store_true", dest="beamf_start",
                      default=False, help="Start network beamformer")
    parser.add_option("--channel-integration-time", action="store", dest="channel_integ",
                      type="float", default=None, help="Integrated channel integration time [default: None]")
    parser.add_option("--beam-integration-time", action="store", dest="beam_integ",
                      type="float", default=None, help="Integrated beam integration time [default: None]")
    parser.add_option("--beamformer-scaling", action="store", dest="beam_scaling",
                      type="int", default=None, help="Beamformer scaling [default: None]")
    parser.add_option("--beam-start_frequency", action="store", dest="start_frequency_channel",
                      type="float", default=None, help="Beamformer scaling [default: None]")
    parser.add_option("--beam-bandwidth", action="store", dest="beam_bandwidth",
                      type="float", default=None, help="Beamformer scaling [default: None]")
    parser.add_option("--port", action="store", dest="port",
                      type="int", default=None, help="Port [default: None]")
    parser.add_option("--lmc_ip", action="store", dest="lmc_ip",
                      default=None, help="IP [default: None]")
    parser.add_option("--lmc_port", action="store", dest="lmc_port",
                      type="int", default=None, help="Port [default: None]")
    parser.add_option("--lmc-mac", action="store", dest="lmc_mac",
                      type="int", default=None, help="LMC MAC address [default: None]")
    parser.add_option("-f", "--bitfile", action="store", dest="bitfile",
                      default=None, help="Bitfile to use (-P still required) [default: None]")
    parser.add_option("-t", "--tiles", action="store", dest="tiles",
                      default=None, help="Tiles to add to station [default: None]")
    parser.add_option("-P", "--program", action="store_true", dest="program",
                      default=False, help="Program FPGAs [default: False]")
    parser.add_option("-I", "--initialise", action="store_true", dest="initialise",
                      default=False, help="Initialise TPM [default: False]")
    parser.add_option("-C", "--program_cpld", action="store_true", dest="program_cpld",
                      default=False, help="Update CPLD firmware (requires -f option)")
    parser.add_option("-T", "--enable-test", action="store_true", dest="enable_test",
                      default=False, help="Enable test pattern (default: False)")
    parser.add_option("--bandwidth", action="store", dest="bandwidth",
                      type="float", default=None, help="Channelizer bandwidth [default: None]")
    parser.add_option("--ddc_frequency", action="store", dest="ddc_frequency",
                      type="float", default=None, help="DDC frequency [default: None]")
    parser.add_option("--sampling_frequency", action="store", dest="sampling_frequency",
                      type="float", default=700e6, help="ADC sampling frequency. Supported frequency are 700e6, 800e6 [default: 700e6]")
    parser.add_option("--saveraw", action="store_true", dest="saveraw",
                      default=False, help="Save HDF5 Raw File")
    parser.add_option("--ylim", action="store", dest="ylim",
                      default="-100,0", help="Y Limits")

    (conf, args) = parser.parse_args(argv[1:])

    # Set logging
    log = logging.getLogger('')
    log.setLevel(logging.INFO)
    line_format = logging.Formatter("%(asctime)s - %(levelname)s - %(threadName)s - %(message)s")
    ch = logging.StreamHandler(stdout)
    ch.setFormatter(line_format)
    log.addHandler(ch)

    # Set current thread name
    threading.currentThread().name = "Station"

    # Load station configuration
    configuration = load_station_configuration(conf)

    # Create station
    station = Station(configuration)

    # Connect station (program, initialise and configure if required)
    station.connect()

    spead_rx_inst = SpeadRx(conf.saveraw)

    resolutions = 2 ** np.array(range(16)) * (800000.0 / 2 ** 15)
    rbw = int(closest(resolutions, conf.resolution))
    avg = 2 ** rbw
    nsamples = int(2 ** 15 / avg)
    RBW = (avg * (400000.0 / 16384.0))
    #asse_x = np.arange(nsamples/2 + 1) * RBW * 0.001
    bw = 43750000
    nfreq = int((DATA_LEN / 2 / avg) + 1)
    rbw = bw / (nfreq - 1)
    fc = station.configuration['observation']['ddc_frequency']
    x1 = fc - (bw / 2.)
    x2 = fc + (bw / 2.)
    asse_x = np.linspace(x1, x2, nfreq)

    if not conf.inputlist == "":
        antenna_list = [int(a) for a in conf.inputlist.split(",")]
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
    plt.ion()
    gs = gridspec.GridSpec(rows, cols, wspace=0.3, hspace=0.7, top=0.9, bottom=0.09, left=0.08, right=0.96)
    fig = plt.figure(figsize=(14, 9), facecolor='w')
    ax = []
    ax_lines = []
    ax_annotations = []
    ymin = int(conf.ylim.split(",")[0])
    ymax = int(conf.ylim.split(",")[1])
    #csfont = {'fontname': 'monospace', 'weight': 'bold'}
    for i, ant in enumerate(antenna_list):
        ax += [fig.add_subplot(gs[i])]
        ax[i].set_title("INPUT-%02d" % ant, fontsize=10)
        ax[i].set_ylim(ymin, ymax)
        ax[i].set_xlim(x1, x2)
        ax[i].set_ylabel("dB", fontsize=7)
        ax[i].set_xlabel("MHz", fontsize=7)
        ticks = asse_x[::int(len(asse_x)/4)]
        tickslabels = ["%3.1f" % (t/1000000.) for t in ticks[1:-1]]

        ax[i].set_xticks(ticks[1:-1])
        ax[i].set_xticklabels(tickslabels, fontsize=7)
        ax[i].set_yticks([-100, -80, -60, -40, -20, 0])
        ax[i].set_yticklabels([-100, -80, -60, -40, -20, 0], fontsize=7)

        l, = ax[i].plot(asse_x[3:], (np.zeros(len(asse_x[3:]))-100), color='b')
        ax_lines += [l]
        a = ax[i].annotate("--- dBm", (asse_x[50], -15), fontsize=8, color='r')#, **csfont)
        ax_annotations += [a]
    fig.show()

    i = 0
    try:
        while True: #i < conf.num:
            station.send_raw_data()
            data = spead_rx_inst.get_raw_data()
            #print("DATA LEN: %d" % len(data))
            if not data == []:
                #data = data[antenna_mapping, :, :].transpose((0, 1, 2))
                #print("ANT LEN %d" % len(data[:, 0, 0]))
                for n, ant in enumerate(antenna_list):
                    spettro, power_rms, adu_rms = calcAVGSpectra(np.array(data[ant-1]), avg)
                    ax_lines[n].set_ydata(spettro[3:])
                    ax_annotations[n].set_text("%3.1f dBm" % power_rms)
                fig.canvas.draw()
                fig.canvas.flush_events()
            i = i + 1
    except KeyboardInterrupt:
        del station
        logging.info("End of process.")





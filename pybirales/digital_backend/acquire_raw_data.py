import time
import math
import socket
import h5py
import numpy as np
import matplotlib.pyplot as plt
from struct import *
from optparse import OptionParser


class SpeadRx:
    def __init__(self, write_hdf5):

        self.write_hdf5 = write_hdf5

        self.nof_signals = 32
        self.nof_fpga = 2
        self.nof_signals_per_fpga = self.nof_signals // self.nof_fpga
        self.data_width = 16
        self.data_byte = self.data_width // 8
        self.byte_per_packet = 1024
        self.word_per_packet = self.byte_per_packet / (self.data_width // 8)
        self.fpga_buffer_size = 64 * 1024
        self.nof_samples = self.fpga_buffer_size // 2
        self.expected_nof_packets = self.nof_signals * (self.fpga_buffer_size / self.byte_per_packet) // 2

        self.data_reassembled = np.zeros((self.nof_signals, self.fpga_buffer_size // 2), dtype=np.int16)
        self.data_buff = None
        self.is_spead = 0
        self.logical_channel_id = 0
        self.packet_counter = 0
        self.payload_length = 0
        self.center_frequency = 0
        self.start_channel_id = 0
        self.capture_mode = 0
        self.timestamp = 0
        self.sync_time = 0
        self.fpga_id = 0
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
        self.line_data = []
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
        self.nof_signals_per_fpga = self.nof_signals // self.nof_fpga
        self.data_width = 16
        self.data_byte = self.data_width // 8
        self.byte_per_packet = self.payload_length
        self.word_per_packet = self.byte_per_packet // (self.data_width // 8)
        self.nof_samples = self.fpga_buffer_size // 2
        self.expected_nof_packets = self.nof_signals * (self.fpga_buffer_size / self.byte_per_packet)
        self.data_reassembled = np.zeros((self.nof_signals, self.nof_samples), dtype=np.int16)

    def write_buff(self, data):
        idx = (self.packet_counter * self.word_per_packet) % (self.fpga_buffer_size // self.data_byte)
        self.data_reassembled[self.start_antenna_id, idx: idx + self.word_per_packet] = data
        self.recv_packets += 1

    def buffer_demux(self):
        if self.nof_included_antenna == 1:
            self.data_buff = self.data_reassembled
        else:
            print("Synchronised RAW data not supported!")
            exit()

    def detect_full_buffer(self):
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
                print("Socket timeout!")
                return None

            if packet_ok:
                self.spead_header_decode(_pkt)

                if self.is_spead and self.capture_mode == 0:  # is a SPEAD packet and contains raw data
                    if self.first_packet == 1:
                        self.set_buffers()
                        self.first_packet = 0
                    self.write_buff(unpack('<' + 'h' * (self.payload_length // self.data_byte), _pkt[self.offset:]))
                    buffer_ready = self.detect_full_buffer()
                    if buffer_ready:
                        self.buffer_demux()
                        if self.write_hdf5:
                            self.hdf5_channel.create_dataset(str(self.timestamp), data=self.data_buff)
                        self.num += 1
                        print("Full buffer received: " + str(self.num))
                        return self.data_buff.tolist()

        self.hdf5_channel.close()

    def plot_raw_data(self, data):
        plt.title("RAW data %d" % self.num)
        for n in range(self.nof_signals):
            plt.plot(data[n][0:512])
        plt.show()


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-p", dest="port", default="4660", help="UDP port")
    parser.add_option("-w", dest="write_hdf5", default=False, action="store_true", help="Write HDF5 files")
    parser.add_option("-d", dest="plot", default=True, action="store_false", help="Generates data plots")

    (options, args) = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
    sock.bind(("10.0.10.201", 4660))
    sock.settimeout(1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)

    spead_rx_inst = SpeadRx(options.write_hdf5)

    while True:
        raw_data = spead_rx_inst.get_raw_data(sock)

        if options.plot and raw_data is not None:
            print(len(raw_data))
            spead_rx_inst.plot_raw_data(raw_data)

from __future__ import division

import inspect
import signal
import time

import numpy as np
import threading
import logging
import socket
import struct

# Stopping flag
stop_acquisition = False


def _signal_handler(signum, frame):
    global stop_acquisition
    logging.info("Received interrupt, stopping acqusition")
    stop_acquisition = True


class ChannelisedData(object):
    """ REACH spectrometer data receiver """

    def __init__(self, ip, port=4660, nof_signals=32, buffer_samples=65536, callback=None):
        """ Class constructor:
        @param ip: IP address to bind receiver to
        @param port: Port to receive data on """

        # Initialise parameters
        self._nof_signals_per_fpga = nof_signals // 2
        self._sampling_time = 1.1702857142857143e-05
        self._buffer_samples = buffer_samples
        self._nof_signals = nof_signals
        self._port = port
        self._ip = ip

        # Create socket reference
        self._socket = None

        # Packet header content
        self._packet_counter = 0
        self._payload_length = 0
        self._sync_time = 0
        self._timestamp = 0
        self._start_antenna_id = 0
        self._offset = 9 * 8

        # Book keeping
        self._reference_counter = 0
        self._rollover_counter = 0
        self._received_packets = 0
        self._reference_time = 0

        # Received data placeholder
        self._receiver_thread = None
        self._received_data = None
        self._current_timestamp = None

        # Placeholder for full buffer
        self._full_buffer = None
        self._full_buffer_timestamp = None

        # Callback (sanity check on number of parameters
        self._callback = None
        if callback is not None:
            try:
                inspect.getargspec(callback)
                self._callback = callback
            except TypeError:
                logging.error("Invalid function callback, ignoring")

    def initialise(self):
        """ Initialise receiver """

        # Initialise socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((self._ip, self._port))
        self._socket.settimeout(2)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)

        # Initialise channel data receiver
        self._received_data = np.empty((self._buffer_samples, self._nof_signals), dtype=np.complex64)

    def receive_channel_data(self):
        """ Wait for a spead packet to arrive """

        # Clear receiver
        self._clear_receiver()

        # Check if receiver has been initialised
        if self._socket is None:
            logging.error("Spectrum receiver not initialised")
            return

        # Loop until required to stop
        while not stop_acquisition:
            # Try to acquire packet
            try:
                packet, _ = self._socket.recvfrom(9000)
            except socket.timeout:
                logging.info("Socket timeout")
                continue

            # We have a packet, check if it is a valid packet
            if not self._decode_spead_header(packet):
                print("Invalid spead packet")
                continue

            # Valid packet, extract payload
            payload = struct.unpack('<' + 'h' * (self._payload_length // 2), packet[self._offset:])

            # Calculate number of samples in packet
            samples_in_packet = self._payload_length // (self._nof_signals_per_fpga * 4)

            # Calculate packet time
            # Sampling time is: 1.0 / (sampling_freq / (DDC (8) * FFT Size (1024)))
            packet_time = self._sync_time + self._timestamp * self._sampling_time

            # Handle packet counter rollover
            # First condition ensures that on startup, first packets with counter number 0 are not updated
            if self._reference_counter == 0:
                self._reference_counter = self._packet_counter
            elif self._packet_counter == 0:
                self._rollover_counter += 1
                self._packet_counter += self._rollover_counter << 24
            else:
                self._packet_counter += self._rollover_counter << 24

            # Set timestamp if first packet
            if self._current_timestamp is None:
                self._current_timestamp = packet_time

            # Calculate packet index
            sample_index = (self._packet_counter - self._reference_counter) % (
                    self._buffer_samples // samples_in_packet)

            # Check if packet belongs to current buffer
            if self._reference_time == 0:
                self._reference_time = packet_time

            # If packet time is less than reference time, then this belongs to the previous buffer
            if packet_time < self._reference_time:
                print("Packet belongs to previous buffer!")
                continue

            # Check if we skipped buffer boundaries, and if so, persist buffer
            if sample_index == 0 and self._start_antenna_id == 0 and \
                    packet_time >= self._reference_time + self._buffer_samples * self._sampling_time:
                self._persist_buffer()
                self._reference_time = self._buffer_samples * self._sampling_time
                self._current_timestamp = packet_time
                self._received_packets = 0

            # Increment number of received packet
            self._received_packets += 1

            # Add data to buffer
            self._add_packet_to_buffer(payload, sample_index * samples_in_packet, samples_in_packet,
                                       self._start_antenna_id, packet_time)

    def _add_packet_to_buffer(self, payload, start_index, samples_in_packet, start_antenna, packet_time):
        """ Add received payload to buffer """

        # Incoming data is in shape time/antennas/coeff (complex coefficient), reshape
        data = np.reshape(payload, (samples_in_packet, self._nof_signals_per_fpga, 2))

        # Convert to complex
        data = data[:, :, 0] + 1j * data[:, :, 1]

        # Convert to complex64
        data = data.astype(np.complex64)

        # Place in container
        self._received_data[start_index: start_index + samples_in_packet,
                            start_antenna: start_antenna + self._nof_signals_per_fpga] = data

        pass

    def _persist_buffer(self):
        """ Buffer is full, send out for processing"""

        # Wait for full buffer to be None
        while self._full_buffer is not None and not stop_acquisition:
            print("Waiting for full buffer to be emptied")
            time.sleep(1)

        print("Persisting buffer with {} packets".format(self._received_packets))

        # Copy full buffer to full buffer placeholder
        self._full_buffer = self._received_data.copy()
        self._full_buffer_timestamp = self._current_timestamp

        # Done, Clear buffer
        self._received_data[:] = 0

        # If a callback is defined, call callback()
        if self._callback is not None:
            self._callback()

    def read_buffer(self):
        """ Wait for full buffer """
        while self._full_buffer is None and not stop_acquisition:
            time.sleep(0.01)

        return self._full_buffer, self._full_buffer_timestamp

    def read_buffer_ready(self):
        """ Ready from buffer read """
        self._full_buffer = None

    def start_receiver(self):
        """ Receive specified number of spectra """

        # Create and start thread and wait for it to stop
        self._receiver_thread = threading.Thread(target=self.receive_channel_data)
        self._receiver_thread.name = "ChannelAcqusition"
        self._receiver_thread.start()

    def stop(self):
        """ Wait for receiver to finish """
        global stop_acquisition

        if self._receiver_thread is None:
            logging.error("Receiver not started")

        # Issue stop
        stop_acquisition = True

        # Wait for thread to finish
        self._receiver_thread.join()

    def _decode_spead_header(self, packet):
        """ Decode SPEAD packet header
        @param: Received packet header """

        # Unpack SPEAD header items
        try:
            items = struct.unpack('>' + 'Q' * 9, packet[0:8 * 9])
        except:
            logging.error("Error processing packet")
            return False

        # Process all spead items
        for idx in range(len(items)):
            spead_item = items[idx]
            spead_id = spead_item >> 48
            val = spead_item & 0x0000FFFFFFFFFFFF
            if spead_id == 0x5304 and idx == 0:
                continue
            elif idx == 0 and not spead_id == 0x5304:
                return False
            elif spead_id == 0x8001:
                heap_counter = val
                self._packet_counter = heap_counter & 0xFFFFFF
            elif spead_id == 0x8004:
                self._payload_length = val
            elif spead_id == 0x9027:
                self._sync_time = val
            elif spead_id == 0x9600:
                self._timestamp = val
            elif spead_id == 0xA004:
                if val & 0xEF != 7:
                    return False
            elif spead_id == 0xA002:
                self._start_antenna_id = (val & 0x000000000000FF00) >> 8
            elif spead_id == 0xA003 or spead_id == 0xA001:
                pass
            elif spead_id == 0x3300:
                pass
            else:
                logging.error("Error in SPEAD header decoding!")
                logging.error("Unexpected item {} at position {}".format(hex(spead_item), " at position " + str(idx)))

        return True

    def _clear_receiver(self):
        """ Reset receiver  """
        self._received_packets = 0
        self._previous_timestamp = 0


if __name__ == "__main__":
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-p", dest="port", default=4660, type=int, help="UDP port (default:4660)")
    parser.add_option("-i", dest="ip", default="10.0.10.10", help="IP (default: 10.0.10.10)")
    (config, args) = parser.parse_args()

    # Note: buffer samples must be a multiple of 20
    receiver = ChannelisedData(ip=config.ip, port=config.port, buffer_samples=262140)
    receiver.initialise()
    receiver.start_receiver()

    # Wait for exit or termination
    signal.signal(signal.SIGINT, _signal_handler)

    from matplotlib import pyplot as plt

    while not stop_acquisition:
        buff, timestamp = receiver.read_buffer()
        receiver.read_buffer_ready()

        # # plt.imshow(np.angle(buff), aspect='auto')
        # plt.plot(np.angle(buff[:, 20]))
        # plt.show()


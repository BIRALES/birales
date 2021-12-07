from __future__ import division

from multiprocessing import Process, Value, shared_memory
from datetime import datetime
from ctypes import c_bool, c_double
import numpy as np
import inspect
import logging
import signal
import socket
import struct
import time


class ChannelisedData(Process):
    """ BIRALES spectrometer data receiver """

    def __init__(self, ip, port=4660, nof_signals=32, buffer_samples=65536):
        """ Class constructor:
        @param ip: IP address to bind receiver to
        @param port: Port to receive data on """

        # Initialise superclass
        Process.__init__(self)

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

        # Received data placeholder
        self._receiver_thread = None
        self._received_data = None
        self._current_timestamp = None

        # Shared memory from where data can be access from other processes
        # _shared_memory_segment is a shared memory segment created internally
        # _ready_buffer_shared is a numpy array in a shared memory segment
        # _read_timestamp_shared is a shared variable
        self._shared_memory_segment = None
        self._ready_buffer_shared = None
        self._ready_timestamp_shared = Value(c_double, 0.0)
        self._ready_buffer_flag = Value(c_bool, False)
        self._stop_acquisition = Value(c_bool, False)

    def initialise(self):
        """ Initialise receiver """

        # Initialise socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((self._ip, self._port))
        self._socket.settimeout(2)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)

        # Initialise channel data receiver
        self._received_data = np.empty((self._buffer_samples, self._nof_signals), dtype=np.complex64)

        # Create shared memory numpy array
        self._shared_memory_segment = shared_memory.SharedMemory(create=True, size=self._received_data.nbytes)

        # Create a NumPy array backed by shared memory
        self._ready_buffer_shared = np.ndarray(self._received_data.shape,
                                               dtype=np.complex64,
                                               buffer=self._shared_memory_segment.buf)

    def run(self):
        """ Wait for a spead packet to arrive """

        def _signal_handler(signum, frame):
            self._stop_acquisition.value = True

        # Set signal handler
        signal.signal(signal.SIGINT, _signal_handler)

        # Clear receiver
        self._clear_receiver()

        # Check if receiver has been initialised
        if self._socket is None:
            logging.error("Spectrum receiver not initialised")
            return

        # Loop until required to stop
        while not self._stop_acquisition.value:
            # Try to acquire packet
            try:
                packet, _ = self._socket.recvfrom(9000)
            except socket.timeout:
                logging.error("Socket timeout")
                continue

            # We have a packet, check if it is a valid packet
            if not self._decode_spead_header(packet):
                logging.warning("Invalid spead packet")
                continue

            # Valid packet, extract payload
            payload = struct.unpack('<' + 'h' * (self._payload_length // 2), packet[self._offset:])

            # Calculate number of samples in packet
            samples_in_packet = self._payload_length // (self._nof_signals_per_fpga * 4)

            # Calculate packet time
            # Sampling time is: 1.0 / (sampling_freq / (DDC (8) * FFT Size (1024)))
            # NOTE: + 1 due to firmware bug
            packet_time = self._sync_time + self._timestamp * self._sampling_time + 1

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

            # Check if we skipped buffer boundaries, and if so, persist buffer
            if sample_index == 0 and self._received_packets > 1:
                self._persist_buffer()
                self._current_timestamp = packet_time
                self._received_packets = 0

            # Increment number of received packet
            self._received_packets += 1

            # Add data to buffer
            self._add_packet_to_buffer(payload, sample_index * samples_in_packet,
                                       samples_in_packet, self._start_antenna_id)

    def _add_packet_to_buffer(self, payload, start_index, samples_in_packet, start_antenna):
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

    def _persist_buffer(self):
        """ Buffer is full, send out for processing"""

        # Wait for full buffer to be None
        while self._ready_buffer_flag.value and not self._stop_acquisition.value:
            logging.warning("Waiting for full buffer to be emptied")
            time.sleep(1)

        if self._stop_acquisition.value:
            return

        logging.info("Persisting buffer with {} packets".format(self._received_packets))

        # Copy full buffer to full buffer placeholder
        self._ready_buffer_shared[:] = self._received_data
        self._ready_timestamp_shared.value = self._current_timestamp
        self._ready_buffer_flag.value = True

        # Done, Clear buffer
        self._received_data[:] = 0

    def read_buffer(self):
        """ Wait for full buffer """

        # If buffer is not ready, sleep for a while and return None. This avoids
        # waiting forever when either the data stream or receiver is stopped
        if not self._ready_buffer_flag.value and not self._stop_acquisition.value:
            time.sleep(0.1)
            return None, None

        return self._ready_buffer_shared, self._ready_timestamp_shared.value

    def read_buffer_ready(self):
        """ Ready from buffer read """
        self._ready_buffer_flag.value = False

    def stop_receiver(self):
        """ Wait for receiver to finish """

        # Issue stop
        self._stop_acquisition.value = True

        # Wait for a while
        # TODO: Perform the below using an atexit registered function
        time.sleep(0.1)

        # Clear shared memory buffer
        if self._shared_memory_segment is not None:
            del self._ready_buffer_shared
            self._shared_memory_segment.close()
            self._shared_memory_segment.unlink()

        # Close socket
        self._socket.close()

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


if __name__ == "__main__":
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-p", dest="port", default=4660, type=int, help="UDP port (default:4660)")
    parser.add_option("-i", dest="ip", default="10.0.10.10", help="IP (default: 10.0.10.10)")
    parser.add_option("-P", dest="plot", default=False, action="store_true", help="Generate Plot [default: False]")
    (config, args) = parser.parse_args()

    # Note: buffer samples must be a multiple of 20
    receiver = ChannelisedData(ip=config.ip, port=config.port, buffer_samples=262140)
    receiver.initialise()
    receiver.start()


    # Wait for exit or termination
    def _signal_handler(signum, frame):
        logging.info("Received interrupt, stopping acqusition")
        print("Stopping handler")
        receiver.stop_receiver()


    signal.signal(signal.SIGINT, _signal_handler)

    from matplotlib import pyplot as plt
    import matplotlib.dates as md

    sampling_time = 1.1702857142857143e-05

    while not receiver._stop_acquisition.value:

        buff, timestamp = None, None
        while buff is None and not receiver._stop_acquisition.value:
            buff, timestamp = receiver.read_buffer()

        if receiver._stop_acquisition.value:
            break

        
        vals = buff.real
        print("\nReal mean value: ", np.mean(vals))
        print("Real minimum: ", np.min(vals))
        print("Real maximum: ", np.max(vals))
        print("Real standard deviation", np.std(vals))

        vals = buff.imag
        print("\nImag mean value: ", np.mean(vals))
        print("Imag minimum: ", np.min(vals))
        print("Imag maximum: ", np.max(vals))
        print("Imag standard deviation", np.std(vals))

        vals = np.abs(buff)
        print("\nMean absolute value: ", np.mean(vals))
        print("Absolute minimum: ", np.min(vals))
        print("Absolute maximum: ", np.max(vals))
        print("Absolute standard deviation", np.std(vals))
        

        # Generate timestamps
        now = time.mktime(time.localtime())
        dates = [datetime.utcfromtimestamp(timestamp + i * sampling_time) for i in np.arange(buff.shape[0])]

        if config.plot:

            plt.xticks(rotation=25)
            ax = plt.gca()
            xfmt = md.DateFormatter('%M:%S')
            ax.xaxis.set_major_formatter(xfmt)

            plt.plot(dates, np.abs(buff[:, 7]))
            plt.xlabel("Sample")
            plt.title("Antenna 0")
            plt.show()

            plt.clf()

            # plt.imshow(buff.real, aspect='auto')
            # plt.title("Real")
            # plt.xlabel("Antenna")
            # plt.ylabel("Sample")
            # plt.colorbar()

            # plt.figure()
            # plt.imshow(buff.imag, aspect='auto')
            # plt.title("Imaginary")
            # plt.xlabel("Antenna")
            # plt.ylabel("Sample")
            # plt.colorbar()

        receiver.read_buffer_ready()

    print("Exiting")

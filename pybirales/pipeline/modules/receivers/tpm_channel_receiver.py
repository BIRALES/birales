from threading import Thread

import numpy as np
import datetime
import logging
import signal
import json

from pybirales import settings
from pybirales.birales_config import BiralesConfig
from pybirales.digital_backend.acquire_channel import ChannelisedData
from pybirales.pipeline.base.definitions import PipelineError, ObservationInfo
from pybirales.pipeline.base.processing_module import Generator
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob
from pybirales.repository.message_broker import broker


# NOTE: Need to change Linux UDP receive buffer size to reduce packet loss
# sudo sysctl -w net.core.rmem_max=26214400


class TPMReceiver(Generator):
    """ Receiver """

    def __init__(self, config, input_blob=None):
        # This module does not need an input block
        self._validate_data_blob(input_blob, valid_blobs=[type(None)])

        # Sanity checks on configuration
        if {'nsamp', 'nants', 'nsubs', 'nbits', 'npols', 'port', 'ip'} - set(config.settings()) != set():
            raise PipelineError("Receiver: Missing keys on configuration "
                                "(nsamp, nants, nsubs, nbits, npols, port, ip)")
        self._nsamp = config.nsamp
        self._nants = config.nants
        self._nsubs = config.nsubs
        self._nbits = config.nbits
        self._npols = config.npols
        self._port = config.port
        self._ip = config.ip

        self._read_count = 0
        self._metrics_poll_freq = 10
        self._metric_channel = 'antenna_metrics'

        # Datatype
        self._datatype = np.complex64

        # Call superclass initialiser
        super(TPMReceiver, self).__init__(config, input_blob)

        # Pointer to tpm receiver process
        self._tpm_receiver = None

        # Pointer to local thread which interacts with receiver process
        self._data_processor = None

        # Processing module name
        self.name = "TPMReceiver"

    def generate_output_blob(self):
        """ Generate output data blob """
        # Generate blob
        return ReceiverBlob(self._config, [('npols', self._npols),
                                           ('nsubs', self._nsubs),
                                           ('nsamp', self._nsamp),
                                           ('nants', self._nants)],
                            datatype=self._datatype)

    def start_generator(self):
        """ Start receiving data """

        # Start TPM data receiver process
        self._tpm_receiver = ChannelisedData(self._ip,
                                             port=self._port,
                                             nof_signals=self._nants,
                                             buffer_samples=self._nsamp)
        self._tpm_receiver.initialise()
        self._tpm_receiver.start()

        # Start local thread which interacts with TPM receiver process
        self._data_processor = Thread(target=self.process_data)
        self._data_processor.name = "TPMDataProcessor"
        self._data_processor.start()

    def stop_receiver(self):
        """ Stop generator """

        # Set stopping flag
        self._stop_module.set()

        # Stop TPM data receiver
        if self._tpm_receiver is not None:
            self._tpm_receiver.stop_receiver()
            self._tpm_receiver = None

    def process_data(self):
        """ Data callback for receiver thread"""

        while not self._stop_module.is_set():

            # Initialise an ObservationInfo object
            obs_info = ObservationInfo()
            obs_info['sampling_time'] = 1.0 / settings.observation.samples_per_second
            obs_info['start_center_frequency'] = settings.observation.start_center_frequency
            obs_info['channel_bandwidth'] = settings.observation.channel_bandwidth
            obs_info['transmitter_frequency'] = settings.observation.transmitter_frequency
            obs_info['nsubs'] = self._nsubs
            obs_info['nsamp'] = self._nsamp
            obs_info['nants'] = self._nants
            obs_info['npols'] = self._npols

            # Get output blob
            output_data = self.request_output_blob()

            # Get buffer
            data, timestamp = None, None
            while not self._stop_module.is_set() and data is None:
                data, timestamp = self._tpm_receiver.read_buffer()

            # If receiver is stopping, break from while loop
            if self._stop_module.is_set():
                self.release_output_blob(obs_info)
                break

            # Set data and timestamp
            output_data[:] = data.reshape((self._nsubs, self._nsamp, self._nants))
            obs_info['timestamp'] = datetime.datetime.utcfromtimestamp(timestamp)

            # Ready from buffer
            self._tpm_receiver.read_buffer_ready()

            # Release output blob
            self.release_output_blob(obs_info)

            logging.info("Receiver: Received buffer ({})".format(obs_info['timestamp'].time()))

            # Publish the RMS voltages
            self.publish_antenna_metrics(self._read_count, output_data, obs_info)

            self._read_count += 1

        # Stop module has been set
        self.stop_receiver()

    @staticmethod
    def _calculate_rms(input_data):
        """ Calculate the RMS of the incoming antenna data
        :param input_data: Input antenna data """
        return np.squeeze(np.sqrt(np.mean(np.power(np.abs(input_data), 2.), axis=2)))

    def publish_antenna_metrics(self, iteration, data, obs_info):
        if iteration % self._metrics_poll_freq == 0:
            rms_voltages = self._calculate_rms(data).tolist()
            timestamp = obs_info['timestamp'].isoformat('T')

            msg = json.dumps({'timestamp': timestamp,
                              'voltages': rms_voltages})
            broker.publish(self._metric_channel, msg)

            logging.debug('Published antenna metrics %s: %s', timestamp,
                          ', '.join(['%0.2f'] * len(rms_voltages)) % tuple(rms_voltages))

            logging.debug('Delay between server and roach is ~ {:0.2f} seconds'.format(
                (datetime.datetime.utcnow() - obs_info['timestamp']).total_seconds()))


if __name__ == "__main__":

    # Load configuration
    BiralesConfig().load()

    # Note: buffer samples must be a multiple of 20
    receiver = TPMReceiver(settings.tpm_receiver)
    receiver.start_generator()

    def _signal_handler(signum, frame):
        receiver.stop_module()

    # Wait for exit or termination
    signal.signal(signal.SIGINT, _signal_handler)

    from time import sleep

    while not receiver.is_stopped:
        sleep(1)

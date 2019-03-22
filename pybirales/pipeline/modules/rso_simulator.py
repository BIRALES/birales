import datetime
import logging

import numpy as np
from scipy.signal import chirp

from pybirales import settings
from pybirales.pipeline.base.definitions import PipelineError, ObservationInfo
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.dummy_data import DummyBlob
from pybirales.pipeline.base.definitions import NoDataReaderException


class RSOSignature():
    def __init__(self, noise_std, track_start, track_length, snr, doppler_shift, doppler_gradient, modulate=True):

        self.id = id(self)
        self.snr = snr
        self.td = 1. / settings.observation.samples_per_second
        self._start_freq = settings.observation.transmitter_frequency * 1e6 - doppler_shift
        self._end_freq = self._start_freq + doppler_gradient * track_length

        self.ts = np.arange(start=self.td * (track_start / self.td), stop=track_start + track_length, step=self.td)

        # Calculate the amplitude needed for the desired SNR
        self.amp = self.calculate_amplitude(noise_std, self.snr)

        # Original signal. The imaginary part needs to be symmetric and complex conjugate to the real part.
        # Otherwise it will be mirrored after channelisation.
        r = self.amp * chirp(t=self.ts, f0=self._start_freq, f1=self._end_freq, t1=track_start + track_length)
        i = self.amp * chirp(t=self.ts, f0=self._start_freq, f1=self._end_freq, t1=track_start + track_length,
                             phi=90) * 1j

        self.signal = r + i

        if modulate:
            # Modulate the signal with a sinc wave
            self.signal *= self.modulate_signal()

        logging.info(
            "Generated RSO {}: Doppler {:0.3f} Hz, Doppler gradient {:0.3f} Hz/s, from T={:0.3f}s till T={:0.3f}s (Modulated: {})".format(
                self.id, doppler_shift, doppler_gradient, track_start, track_start + track_length, modulate))

    def calculate_amplitude(self, noise_std, snr):
        """

        :param noise_std:
        :param snr:
        :return:
        """
        noise_avg_watts = noise_std ** 2
        noise_avg_db = 10 * np.log10(noise_avg_watts)

        signal_avg_db = noise_avg_db + snr
        sig_avg_watts = 10 ** (signal_avg_db / 10)

        # we can use this as amplitude for our chirp
        return np.sqrt(sig_avg_watts)

    def modulate_signal(self):
        """

        :return:
        """
        a = len(self.ts) * self.td
        ts1 = np.linspace(start=-a / 2, stop=a / 2., num=len(self.ts))

        return np.sinc(ts1)

    def get_signal(self, t0, t1):
        """

        :param t0:
        :param t1:
        :return:
        """

        t_min = np.min(self.ts)
        if t0 < t_min:
            t0 = t_min

        t_max = np.max(self.ts)
        if t1 >= t_max:
            t1 = t_max

        i = np.where(np.bitwise_and(self.ts < t1, self.ts >= t0))

        if np.any(i):
            samples = np.shape(i)[1]

            if t0 == t_min:
                start = settings.rso_generator.nsamp - samples
                stop = settings.rso_generator.nsamp
            elif t1 == t_max:
                start = 0
                stop = samples
            else:
                start = 0
                stop = settings.rso_generator.nsamp

            return start, stop, self.signal[i]

        return None, None, None


class RSOGenerator(ProcessingModule):

    def __init__(self, config, input_blob=None):

        # This module does not need an input_blob
        self._validate_data_blob(input_blob, valid_blobs=[type(None)])

        np.random.seed(100)

        # Sanity checks on configuration
        if {'nants', 'nsamp', 'nsubs', 'npols', 'nbits', 'complex'} - set(config.settings()) != set():
            raise PipelineError("DummyDataGenerator: Missing keys on configuration "
                                "(nants, nsamp, nsub, 'npols', 'nbits', complex")
        self._nants = config.nants
        self._nsamp = config.nsamp
        self._nsubs = config.nsubs
        self._npols = config.npols
        self._nbits = config.nbits
        self._complex = config.complex

        # Define data type
        if self._nbits == settings.rso_generator.nbits and self._complex:
            self._datatype = np.complex64
        else:
            raise PipelineError("DummyDataGenerator: Unsupported data type (bits, complex)")

        # Call superclass initialiser
        super(RSOGenerator, self).__init__(config, input_blob)

        self._counter = 0

        # Processing module name
        self.name = "RSO Generator"

        self.mean_noise_power = settings.rso_generator.mean_noise_power

        self.rso_targets = self.generate_rso(self.mean_noise_power)

        self.stop_time = settings.observation.duration + 10

    def generate_rso(self, mean_noise_power):
        tracks = []
        config = settings.rso_generator
        num_rso = int(settings.observation.duration / config.rso_freq)
        max_start_time = settings.observation.duration - config.track_length_range[1]

        for _ in np.arange(0, num_rso):
            track_start = np.random.uniform(3, max_start_time)

            track_length = np.random.uniform(config.track_length_range[0],
                                             config.track_length_range[1])

            snr = np.random.uniform(config.snr_range[0], config.snr_range[1])
            ds = np.random.uniform(config.doppler_range[0], config.doppler_range[1])
            dg = np.random.uniform(config.doppler_gradient_range[0],
                                   config.doppler_gradient_range[1])
            tracks.append(
                RSOSignature(noise_std=mean_noise_power, track_start=track_start, track_length=track_length, snr=snr,
                             doppler_shift=ds,
                             doppler_gradient=dg))

        if config.tx_snr > 0:
            tx = RSOSignature(noise_std=mean_noise_power, track_start=0.5, track_length=settings.observation.duration,
                              snr=config.tx_snr,
                              doppler_shift=0,
                              doppler_gradient=0, modulate=False)

            tracks.append(tx)

        return tracks

    def generate_noise(self, mean_noise_power):
        n_samples = self._npols * self._nsubs * self._nsamp * self._nants

        noise = np.random.normal(0, mean_noise_power, n_samples) + 1j * np.random.normal(0, mean_noise_power, n_samples)

        noise = noise.reshape((self._npols, self._nsubs, self._nsamp, self._nants))
        return noise.astype(self._datatype)

    def process(self, obs_info, input_data, output_data):
        """

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        step = 1. / settings.observation.samples_per_second
        t0 = self._counter * self._nsamp * step
        t1 = (self._counter * self._nsamp + self._nsamp) * step

        if t0 > self.stop_time:
            # Stop the RSO generator if observation duration (+ a small buffer) is exceeded
            raise NoDataReaderException(
                "RSO Generator generated {} seconds of data. Stopping observation".format(self.stop_time))

        data = np.zeros((self._npols, self._nsubs, self._nsamp, self._nants), dtype=self._datatype)

        # add rso targets
        for rso in self.rso_targets:
            start, stop, signal = rso.get_signal(t0=t0, t1=t1)

            if np.any(signal):
                data[:, :, start:stop, 15] += signal

        output_data[:] = data + self.generate_noise(mean_noise_power=self.mean_noise_power)

        self._counter += 1

        # Create observation information
        obs_info = ObservationInfo()
        obs_info['sampling_time'] = 1. / settings.observation.samples_per_second
        obs_info['timestamp'] = datetime.datetime.utcnow()
        obs_info['nsubs'] = self._nsubs
        obs_info['nsamp'] = self._nsamp
        obs_info['nants'] = self._nants
        obs_info['npols'] = self._npols
        obs_info['npols'] = self._npols
        obs_info['channel_bandwidth'] = settings.observation.channel_bandwidth
        obs_info['transmitter_frequency'] = settings.observation.transmitter_frequency
        obs_info['start_center_frequency'] = settings.observation.start_center_frequency

        obs_info['rso_tracks'] = self.rso_targets

        logging.debug("Input data: %s", np.sum(input_data))
        logging.debug("Output data: %s shape: %s", np.sum(output_data), output_data.shape)

        return obs_info

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return DummyBlob(self._config, [('npols', self._npols),
                                        ('nsubs', self._nsubs),
                                        ('nsamp', self._nsamp),
                                        ('nants', self._nants)],
                         datatype=self._datatype)

    def generate_corrdata(self):

        return np.ones((1, 1, 32, 70000), dtype=np.complex64)

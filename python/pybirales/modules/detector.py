import numpy as np
from functools import partial

from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule
from pybirales.blobs.channelised_data import ChannelisedBlob
from pybirales.modules.detection.queue import BeamCandidatesQueue
from pybirales.modules.detection.repository import BeamCandidateRepository
from pybirales.modules.detection.repository import ConfigurationRepository
from pybirales.modules.detection.strategies.m_dbscan import m_detect
from multiprocessing import Pool

from pybirales.base import settings
from pybirales.modules.detection.beam import Beam


class Detector(ProcessingModule):
    def __init__(self, config, input_blob=None):
        if type(input_blob) is not ChannelisedBlob:
            raise PipelineError(
                "Detector: Invalid input data type, should be BeamformedBlob, DummyBlob or ReceiverBlob")

        # Repository Layer for saving the configuration to the Data store
        self._configurations_repository = ConfigurationRepository()

        # Data structure that hold the detected debris (for merging)
        self._debris_queue = BeamCandidatesQueue(settings.beamformer.nbeams)

        # Flag that indicates whether the configuration was persisted
        self._config_persisted = False

        self.pool = Pool(12)

        self.counter = 0

        self.noise = []

        self.mean_noise = 0

        self.channels = None

        self.time = None

        self._doppler_mask = None

        super(Detector, self).__init__(config, input_blob)

        self.name = "Detector"

    def _get_noise_estimation(self, input_data):
        if self.counter < settings.detection.n_noise_samples:
            power = np.power(np.abs(input_data[0, :, settings.detection.noise_channels, :]), 2)

            if settings.detection.noise_use_rms:
                #  use RMS
                noise = np.sqrt(np.mean(np.power(power, 2)))
            else:
                # use mean
                noise = np.mean(power)

            self.noise.append(noise)
            self.mean_noise = np.mean(self.noise)
        return float(self.mean_noise)

    def _get_doppler_mask(self, channels, obs_info):
        if self._doppler_mask is None:
            a = obs_info['transmitter_frequency'] + settings.detection.doppler_range[0] * 1e-6
            b = obs_info['transmitter_frequency'] + settings.detection.doppler_range[1] * 1e-6

            self._doppler_mask = np.bitwise_and(channels < b, channels > a)

        return self._doppler_mask

    def _get_channels(self, obs_info):
        if self.channels is None:
            self.channels = np.arange(obs_info['start_center_frequency'],
                                      obs_info['start_center_frequency'] + obs_info['channel_bandwidth'] * obs_info[
                                          'nchans'],
                                      obs_info['channel_bandwidth'])
            self.channels = self.channels[self._get_doppler_mask(self.channels, obs_info)]

        return self.channels

    def _get_time(self, obs_info):
        if self.time is None:
            self.time = np.arange(0, obs_info['nsamp'])
        return self.time

    def process(self, obs_info, input_data, output_data):
        """
        Run the Space Debris Detector pipeline
        :return void
        """
        channels = self._get_channels(obs_info)
        time = self._get_time(obs_info)
        doppler_mask = self._get_doppler_mask(channels, obs_info)

        # estimate the noise from the data
        obs_info['noise'] = self._get_noise_estimation(input_data)

        if settings.detection.doppler_subset:
            input_data = input_data[:, :, doppler_mask, :]

        if not self._config_persisted:
            self._configurations_repository.persist(obs_info)
            self._config_persisted = True

        beam_candidates = []
        beams = [Beam(beam_id=n_beam,
                      obs_info=obs_info,
                      channels=channels,
                      time=time,
                      beam_data=input_data)
                 for n_beam in range(settings.detection.beam_range[0], settings.detection.beam_range[1])]

        if settings.detection.multi_proc:
            func = partial(m_detect, obs_info, self._debris_queue)
            beam_candidates = self.pool.map(func, beams)

        self._debris_queue.set_candidates(beam_candidates)

        self.counter += 1

    def generate_output_blob(self):
        pass

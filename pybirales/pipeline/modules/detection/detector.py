import numpy as np
import os
import logging as log

from functools import partial
from multiprocessing import Pool

from pybirales import settings
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.pipeline.modules.detection.beam import Beam
from pybirales.pipeline.modules.detection.dbscan_detection import detect
from pybirales.pipeline.modules.detection.queue import BeamCandidatesQueue
from pybirales.repository.models import Observation


class Detector(ProcessingModule):
    _valid_input_blobs = [ChannelisedBlob]

    def __init__(self, config, input_blob=None):
        # Ensure that the input blob is of the expected format
        self._validate_data_blob(input_blob, valid_blobs=[ChannelisedBlob])

        # Data structure that hold the detected debris (for merging)
        self._debris_queue = BeamCandidatesQueue(settings.beamformer.nbeams)

        # Flag that indicates whether the configuration was persisted
        self._config_persisted = False

        self.pool = Pool(settings.detection.n_procs)

        self.counter = 0

        self.noise = []

        self.mean_noise = 0

        self.channels = None

        self.time = None

        self._doppler_mask = None

        self._observation = None

        super(Detector, self).__init__(config, input_blob)

        self.name = "Detector"

        if settings.detection.visualise_beams:
            log.debug('Create fits directory for observation')
            os.makedirs(os.path.join(os.environ['HOME'],
                                     settings.detection.visualise_fits_dir,
                                     settings.observation.name))

    def _tear_down(self):
        self.pool.close()

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

    def _get_doppler_mask(self, tx, channels):
        if self._doppler_mask is None:
            a = tx + settings.detection.doppler_range[0] * 1e-6
            b = tx + settings.detection.doppler_range[1] * 1e-6

            self._doppler_mask = np.bitwise_and(channels < b, channels > a)

        return self._doppler_mask

    def _get_channels(self, obs_info):
        if self.channels is None:
            self.channels = np.arange(obs_info['start_center_frequency'],
                                      obs_info['start_center_frequency'] + obs_info['channel_bandwidth'] * obs_info[
                                          'nchans'],
                                      obs_info['channel_bandwidth'])
            self.channels = self.channels[self._get_doppler_mask(obs_info['transmitter_frequency'], self.channels)]

        return self.channels

    def _get_time(self, obs_info):
        if self.time is None:
            self.time = np.arange(0, obs_info['nsamp'])
        return self.time

    def process(self, obs_info, input_data, output_data):
        """
        Run the Space Debris Detector pipeline

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        channels = self._get_channels(obs_info)
        time = self._get_time(obs_info)
        doppler_mask = self._get_doppler_mask(obs_info['transmitter_frequency'], channels)

        # estimate the noise from the data
        obs_info['noise'] = self._get_noise_estimation(input_data)

        if settings.detection.doppler_subset:
            input_data = input_data[:, :, doppler_mask, :]

        # If the configuration was not saved AND the number of noise samples is sufficient, save the noise value.
        if not self._config_persisted and self.counter >= settings.detection.n_noise_samples:
            self._observation = Observation.objects.get(id=settings.observation.id)
            self._observation.noise_estimate = self._get_noise_estimation(input_data)
            self._observation.save()
            self._config_persisted = True

        beams = [Beam(beam_id=n_beam,
                      obs_info=obs_info,
                      channels=channels,
                      time=time,
                      beam_data=input_data)
                 for n_beam in range(settings.detection.beam_range[0], settings.detection.beam_range[1])]

        if settings.detection.multi_proc:
            func = partial(detect, obs_info, self._debris_queue)
            beam_candidates = self.pool.map(func, beams)
        else:
            beam_candidates = [detect(obs_info, self._debris_queue, beam) for beam in beams]

        self._debris_queue.set_candidates(beam_candidates)

        self.counter += 1

        return obs_info

    def generate_output_blob(self):
        pass

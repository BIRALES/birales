import logging as log
import os
from functools import partial
from multiprocessing import Pool

import numpy as np

from pybirales import settings
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.pipeline.modules.detection.beam import Beam
from pybirales.pipeline.modules.detection.dbscan_detection import detect
from pybirales.pipeline.modules.detection.queue import BeamCandidatesQueue


class Detector(ProcessingModule):
    _valid_input_blobs = [ChannelisedBlob]

    def __init__(self, config, input_blob=None):
        # Ensure that the input blob is of the expected format
        self._validate_data_blob(input_blob, valid_blobs=[ChannelisedBlob])

        # Data structure that hold the detected debris (for merging)
        self._debris_queue = BeamCandidatesQueue(settings.beamformer.nbeams)

        # Flag that indicates whether the configuration was persisted
        self._config_persisted = False

        self.pool = None
        if settings.detection.multi_proc:
            self.pool = Pool(settings.detection.n_procs)

        self.counter = 0

        self.channels = None

        self.time = None

        self._doppler_mask = None

        super(Detector, self).__init__(config, input_blob)

        self.name = "Detector"

        self._candidates_detected = 0

    def _tear_down(self):
        self.pool.close()

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

            if settings.detection.doppler_subset:
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
        print obs_info['timestamp']
        channels = self._get_channels(obs_info)
        time = self._get_time(obs_info)
        doppler_mask = self._get_doppler_mask(obs_info['transmitter_frequency'], channels)

        if settings.detection.doppler_subset:
            input_data = input_data[:, doppler_mask, :]

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
            beam_candidates = []
            for beam in beams:
                beam_candidates.append(detect(obs_info, self._debris_queue, beam))

        self._candidates_detected += np.count_nonzero(beam_candidates)

        log.info('Result: Detected {} beam candidates across {} beams'.format(self._candidates_detected, len(beams)))

        self._debris_queue.set_candidates(beam_candidates)

        self.counter += 1

        return obs_info

    def generate_output_blob(self):
        pass

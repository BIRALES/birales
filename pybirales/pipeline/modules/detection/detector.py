import logging as log
import Queue
from functools import partial
from multiprocessing import Pool

import numpy as np

from pybirales import settings
from pybirales.events.events import SpaceDebrisDetectedEvent
from pybirales.events.publisher import EventsPublisher
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.pipeline.modules.detection.beam import Beam
from pybirales.pipeline.modules.detection.dbscan_detection import detect
from pybirales.pipeline.modules.detection.queue import BeamCandidatesQueue
from pybirales.pipeline.modules.detection.space_debris_candidate import SpaceDebrisTrack
from pybirales.pipeline.modules.detection.tdm.tdm import TDMWriter


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

        self._publisher = EventsPublisher.Instance()

        # A list of space debris tracks
        self._candidates = []

        self._tdm_writer = TDMWriter(queue=Queue.Queue())

        self._tdm_writer.start()

        self._detection_counter = 0

        # Write to disk every N iterations
        self._write_freq = 10

    def _tear_down(self):
        self._tdm_writer.stop()
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
        self.time = np.arange(0, obs_info['nsamp']) + self.counter * obs_info['nsamp']

        return self.time

    def process(self, obs_info, input_data, output_data):
        """
        Run the Space Debris Detector pipeline

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """
        obs_info['iter_count'] = self.counter
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

        # Create Space Debris Tracks from tracks in beams
        for beam_candidate_list in beam_candidates:
            for beam_candidate in beam_candidate_list:
                for candidate in self._candidates:
                    # If beam candidate is similar to candidate, merge it.
                    if candidate.is_parent_of(beam_candidate) and beam_candidate.is_linear and beam_candidate.is_valid:
                        log.debug(
                            'Beam candidate {} (m={}, c={}, s={}, n={}) added to track {}'.format(id(beam_candidate),
                                                                                                  beam_candidate.m,
                                                                                                  beam_candidate.c,
                                                                                                  beam_candidate.score,
                                                                                                  len(
                                                                                                      beam_candidate.time_data),
                                                                                                  id(candidate)))

                        candidate.add(beam_candidate)
                        break
                else:
                    # Beam candidate does not match any candidate. Create candidate from it.
                    if beam_candidate.is_linear and beam_candidate.is_valid:
                        # Transform this beam candidate into a space debris track
                        self._detection_counter += 1
                        sd = SpaceDebrisTrack(det_no=self._detection_counter, obs_info=obs_info,
                                              beam_candidate=beam_candidate)

                        log.debug('Created new track {} from Beam candidate {}'.format(id(sd), id(beam_candidate),
                                                                                       beam_candidate.debug_msg))
                        # Publish event: A space debris detection was made
                        self._publisher.publish(SpaceDebrisDetectedEvent(sd))

                        # Add the space debris track to the candidates list
                        self._candidates.append(sd)

        temp_candidates = []
        for c in self._candidates:
            if c.is_finished(current_time=self.last_time_sample, min_channel=channels[0], iter_count=self.counter):
                log.debug('Track {} (n: {}) has transitted outside detection window. Removing it from candidates list'
                          .format(id(c), c.size))
            else:
                log.debug('Track {} (n: {}) is still within detection window'.format(id(c), c.size))
                temp_candidates.append(c)

        log.info('Result: {} have space debris tracks have transitted. {} currently in detection window.'.format(
            len(self._candidates) - len(temp_candidates), len(self._candidates)))
        self._candidates = temp_candidates

        self.counter += 1

        if self.counter % self._write_freq == -1:
            for c in self._candidates:
                self._tdm_writer.queue.put((c, obs_info))

        return obs_info

    @property
    def last_time_sample(self):
        return self.time[0]

    def generate_output_blob(self):
        pass

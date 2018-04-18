import logging as log

import pandas as pd

from pybirales import settings
from pybirales.repository.models import SpaceDebrisTrack
from pybirales.services.post_processing.writer import TDMWriter, DebugCandidatesWriter


class PostProcessor:
    """
    Module to post-process an observation
    """

    def __init__(self):
        """


        """
        self.remove_duplicate_epoch = settings.detection.select_highest_snr
        self.remove_duplicate_channel = True

        self._tdm_writer = TDMWriter()
        self._debug_writer = DebugCandidatesWriter()

    def _get_candidates(self, observation):
        """
        Get candidates from the database and convert them back to space debris candidates
        :param observation:
        :return:
        """

        detected_candidates = SpaceDebrisTrack.get(observation_id=observation.id)

        tracks = []
        for candidate in detected_candidates:
            candidate.data = pd.DataFrame(data=candidate.data,
                                          columns=['time_sample', 'channel_sample', 'time', 'channel', 'snr',
                                                   'beam_id'])

            if self.remove_duplicate_epoch:
                candidate.data = candidate.data.sort_values('snr', ascending=False).drop_duplicates(
                    'time_sample').sort_index()

            if self.remove_duplicate_channel:
                candidate.data = candidate.data.sort_values('snr', ascending=False).drop_duplicates(
                    'channel_sample').sort_index()

            tracks.append(candidate)

        return detected_candidates

    def process(self, observation):
        """
        Retrieve the candidates detected in this observation and generate the TDM and/or Debug files
        :param observation:
        :return:
        """
        candidates = self._get_candidates(observation)

        if not candidates:
            log.warning(
                'No candidates were found in observation {} (id:{})'.format(observation.name, observation.id))

        if settings.detection.save_tdm:
            for i, candidate in enumerate(candidates):
                self._tdm_writer.write(observation, candidate, i + 1)

        if settings.detection.debug_candidates:
            for i, candidate in enumerate(candidates):
                self._debug_writer.write(observation, candidate, i + 1)

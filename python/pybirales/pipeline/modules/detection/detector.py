from multiprocessing import Pool

from pybirales import settings
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.pipeline.modules.detection.dbscan_detection import detect
from pybirales.pipeline.modules.detection.util import *


class Detector(ProcessingModule):
    _valid_input_blobs = [ChannelisedBlob]

    def __init__(self, config, input_blob=None):
        # Ensure that the input blob is of the expected format
        self._validate_data_blob(input_blob, valid_blobs=[ChannelisedBlob])

        self.pool = Pool(4)

        self.channels = None

        self._doppler_mask = None

        super(Detector, self).__init__(config, input_blob)

        self.name = "Detector"

        # A list of space debris tracks currently in memory
        self._candidates = []

        self._candidates_msds = []

        # Number of RSO tracks detected in this observation
        self._n_rso = 0

        self.detections = {}

    def _tear_down(self):
        """

        :return:
        """
        # if settings.detection.multi_proc:
        self.pool.close()

    def __process_dbscan(self, input_data, obs_info, channels):
        t0 = np.datetime64(obs_info['timestamp'])
        td = np.timedelta64(int(obs_info['sampling_time'] * 1e9), 'ns')
        # [Feature Extraction] Process the input data and identify the detection clusters
        clusters = []
        for beam_id in range(0, 32):
            clusters.extend(detect(input_data, channels, t0, td, self._iter_count, obs_info['channel_noise'], beam_id))

        log.debug('Found {} new candidates in {} beams'.format(len(clusters), 32))

        # [Track Association] Create new tracks from clusters or merge clusters into existing tracks
        candidates = data_association(self._candidates, clusters, obs_info,
                                      notifications=settings.observation.notifications,
                                      save_candidates=settings.detection.save_candidates)

        # [Track Termination] Check each track and determine if the detection object has transitted outside FoV
        self._candidates = active_tracks(obs_info, candidates, self._iter_count)

        # Output a TDM for the tracks that have transitted outside the telescope's FoV
        obs_info['transitted_tracks'] = [c for c in candidates if c not in self._candidates]

        for i, candidate in enumerate(self._candidates):
            log.info("DBSCAN RSO %d: %s", self._n_rso + 1 + i, candidate.state_str())

        return obs_info

    def process(self, obs_info, input_data, output_data):
        """
        Sieve through the pre-processed channelised data for space debris candidates

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """
        # Skip the first few blobs (to allow for an accurate noise estimation to be determined)
        if self._iter_count < 2:
            return obs_info

        obs_info['track_name'] = settings.observation.target_name
        obs_info['iter_count'] = self._iter_count
        obs_info['transitted_tracks'] = []
        self.channels, self._doppler_mask = apply_doppler_mask(self._doppler_mask, self.channels,
                                                               settings.detection.doppler_range,
                                                               obs_info)

        obs_info['doppler_mask'] = self._doppler_mask

        # obs_info = self.__process_dbscan(input_data[:, self._doppler_mask, :], obs_info, self.channels)
        obs_info = self.__process_dbscan(input_data, obs_info, self.channels)

        return obs_info

    def generate_output_blob(self):
        return ChannelisedBlob(self._input.shape, datatype=float)

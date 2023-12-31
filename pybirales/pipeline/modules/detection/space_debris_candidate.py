# -*- coding: utf-8 -*-

import datetime
import logging as log

import numpy as np
import pandas as pd
from mongoengine import *
from scipy import stats
from sklearn import linear_model

from pybirales import settings
from pybirales.events.events import TrackCreatedEvent, TrackModifiedEvent
from pybirales.events.publisher import EventsPublisher
from pybirales.repository.models import SpaceDebrisTrack as SpaceDebrisTrackModel

_linear_model = linear_model.RANSACRegressor(linear_model.LinearRegression())
_db_model = SpaceDebrisTrackModel


def missing_score(param):
    missing = np.setxor1d(np.arange(min(param), max(param)), param)
    return len(missing) / float(len(param))


class Target:
    """
    Describes the target object that a space debris track
    will be associated with
    """

    def __init__(self):
        """
        Describes the instantaneous attributes that should be measured at the transit time

        """
        self.id = None
        self.name = None
        self.transit_time = None
        self.doppler_shift = None
        self.altitude = None
        self.velocity = None
        self.RA = None
        self.DEC = None


class SpaceDebrisTrack:
    def __init__(self, obs_info, cluster=None):
        """

        :param obs_info:
        :param cluster:
        """

        self._id = None  # model id in the database
        self.name = None
        self._created_at = datetime.datetime.utcnow()
        self._obs_info = obs_info
        self._obs_id = settings.observation.id

        self._linear_model = _linear_model
        self._publisher = EventsPublisher.Instance()

        # The attributes of the linear model
        self.m = None
        self.intercept = None
        self.r_value = 0
        self._prev_size = 0

        # The radar cross section of the track
        self._rcs = None

        # The target we are trying to track
        self._target = None

        # The target which most closely matches the parameters
        self._detected_target = None

        # DataFrame encapsulating the track data (snr, time, channel, beam id)
        self.data = pd.DataFrame(columns=['time_sample', 'channel_sample', 'time', 'channel', 'snr', 'beam_id'])

        # The median doppler shift and corresponding timestamp (to be used as reference)
        self.ref_data = {'doppler': 0.0, 'time': None, 'gradient': 0.0, 'snr': 0.0}

        self.m_timestamp = None

        # The time at which the track is expected to exit the detection window
        self._exit_time = None

        # The last iteration counter
        self._last_iter_count = 0

        self._max_candidate_iter = 2

        # The iteration number
        self._iter = 0

        # Is the track saved to the database?
        self._to_save = True

        # Flag, to mark if a notification was sent
        self._notification_sent = False

        self.cancelled = False
        self.terminated = False

        # If a beam cluster is given, add it on initialisation
        valid = self.associate(cluster)

        if not valid:
            self.cancel()

    def cancel(self):
        self.cancelled = True
        self.terminated = False

    def terminate(self):
        self.cancelled = False
        self.terminated = True

    @property
    def id(self):
        return id(self)

    def send_notification(self):
        """
        Publish a new notification for this track
        :return:
        """

        if self._prev_size == 0:
            # Send a space debris track was created event
            self._publisher.publish(TrackCreatedEvent(self))

        elif (self.size - self._prev_size) / self.size > 0.10:
            # Only send a notification when the track grew by 10% (to avoid lots of messages)
            self._publisher.publish(TrackModifiedEvent(self))

        self._prev_size = self.size

    @property
    def saved(self):
        return not self._to_save

    @property
    def duration(self):
        return self.data['time'].max() - self.data['time'].min()

    @property
    def size(self):
        return len(np.unique(self.data['time_sample']))

    @property
    def beam_size(self):
        return len(np.unique(self.data[['beam_id', 'time_sample']].values))

    @property
    def activated_beams(self):
        return self.data['beam_id'].unique().size

    def _update(self, new_df):
        self.data = new_df
        self.m, self.intercept, self.r_value, self.p_value, self.std_err = self.linear_model(
            self.data['channel_sample'], self.data['time_sample'])

        self._last_iter_count = self.data['iter'].max()

        self.data = self.data.sort_values(by='time')

        mid = int(self.data['channel'].size / 2)
        self.ref_data['time'] = self.data.iloc[mid]['time']
        self.ref_data['time_sample'] = self.data.iloc[mid]['time_sample']
        self.ref_data['channel'] = self.data.iloc[mid]['channel']
        self.ref_data['channel_sample'] = self.data.iloc[mid]['channel_sample']
        self.ref_data['doppler'] = (self.data.iloc[mid]['channel'] - self._obs_info['transmitter_frequency']) * 1e6
        self.ref_data['gradient'] = self.m
        self.ref_data['snr'] = float(np.mean(self.data['snr']))
        self.ref_data['psnr'] = float(np.max(self.data['snr']))
        self._to_save = True

    def state_str(self):
        return "df:{:3.2f} Hz at {:%H:%M:%S on %d.%m.%y} across {} beams\n" \
               "\t\t\t\t\t\t- INFO - Score: {:0.3f}, N: {} ({} unique), SNR: {:2.2f} dB (PSNR:{:2.2f} dB) \n" \
               "\t\t\t\t\t\t- INFO - df/dt:{:3.2f} Hz/s. Beams: {}, C:{:3.2f}, T:{:3.2f}, TN:{:3.2f}".format(
            self.ref_data['doppler'],
            self.ref_data['time'],
            self.activated_beams,
            self.r_value,
            len(self.data['beam_id']),
            self.size,
            self.ref_data['snr'],
            self.ref_data['psnr'],
            self.ref_data['gradient'],
            str(self.data['beam_id'].unique()[:5]),
            np.mean(self.data['channel_sample']),
            np.mean(self.data['time_sample']),
            np.max(self.data['time_sample'])
        )

    def associate(self, cluster_df):
        """
        Associate a beam candidate with this space debris track.
        Update the space debris track's attributes based on new data

        :param cluster_df:
        :type cluster_df: ndarray
        :return:
        """

        def _merge_tmp_df(df, delta_df):
            """
            Create a temporary data frame (merge track df with cluster df)
            :param df: The data frame of the track
            :param delta_df: The cluster df that is to be merged
            :return:
            """

            if df.empty:
                return delta_df
            return pd.concat([self.data, cluster_df])

        def _is_tmp_valid(candidate):
            score = missing_score(candidate['time_sample'])
            g = 1
            if score > 1:
                log.debug("Candidate {}, dropped since missing score is not high enough ({:0.3f})".format(g, score))
                return False, None

            m, c, r_value, p, _ = self.linear_model(candidate['channel_sample'], candidate['time_sample'])

            if r_value > -.99:
                log.debug("Candidate {}, dropped since r-value is not high enough ({:0.3f})".format(g, r_value))
            elif p > 0.01:
                log.debug("Candidate {}, dropped since p-value is not low enough ({:0.3f})".format(g, p))
            else:

                if settings.detection.gradient_thold[0] >= m >= settings.detection.gradient_thold[1]:
                    return True, candidate

            return False, None

        tmp_merged_df = _merge_tmp_df(self.data, cluster_df)

        is_valid, new_data = _is_tmp_valid(tmp_merged_df)

        if is_valid:
            self._update(new_data)

        return is_valid

    def is_valid(self):
        """
        The candidate is not valid if size is too small or the number of activate beams is less than 2
        :return:
        """
        min_beams = 1
        unq_samples = 5
        m_score = 0.25

        # Never delete an object that was detected a minimum number of beams
        if self.data['beam_id'].unique().size > 5:
            return True, None

        # Check that the number of beams activated is less than 2
        if self.data['beam_id'].unique().size < min_beams:
            return False, 'Not enough unique beams {}'.format(id(self) % 1000)

        # Check that the candidate has the minimum number of unique channel and time data points
        if self.data['channel'].unique().size < unq_samples or self.data['time'].unique().size < unq_samples:
            return False, 'Not enough unique samples {}'.format(id(self) % 1000)

        if not np.abs(self.r_value) >= settings.detection.linearity_thold:
            return False, 'Track is not linear {}'.format(id(self) % 1000)

        g_thold = settings.detection.gradient_thold
        if not g_thold[0] >= self.ref_data['gradient'] >= g_thold[1]:
            return False, 'Gradient is not within valid range {}'.format(id(self) % 1000)

        if missing_score(self.data['time_sample']) > m_score:
            return False, 'High missing score {:0.3f} {}'.format(missing_score(self.data['time_sample']),
                                                                 id(self) % 1000)

        return True, None

    def is_parent_of(self, cluster):
        """
        Determine whether a detection cluster should be associated with this space debris track by giving
        a similarity score. Similarity between beam candidate and space debris track is determined by
        cosine distance.

        :param detection_cluster: The detection cluster with which the track is being compared
        :return:
        """
        m_thold = settings.detection.gradient_thold
        track_data = self.aggregate_data(self.data, remove_duplicate_epoch=True, remove_duplicate_channel=True)
        cluster_data = self.aggregate_data(cluster, remove_duplicate_epoch=True, remove_duplicate_channel=True)

        cluster_m, c1, _, _, _ = self.linear_model(cluster_data['channel_sample'], cluster_data['time_sample'])
        track_m, c2, _, _, _ = self.linear_model(track_data['channel_sample'], track_data['time_sample'])
        m2, c3, _, _, e = self.linear_model(pd.concat([track_data['channel_sample'], cluster_data['channel_sample']]),
                                            pd.concat([track_data['time_sample'], cluster_data['time_sample']]))

        v = np.array([track_m, cluster_m, m2])
        c = np.array([c1, c2, c3])

        slope_diff = np.abs(np.std(v) / np.mean(v))
        i_diff = np.abs(np.std(c) / np.mean(c))

        thres = settings.detection.similarity_thold
        is_parent = slope_diff < thres and i_diff < thres and m_thold[1] <= m2 <= m_thold[0]

        # log.debug(
        #     "Track {}: {:2.3f}. {:2.3f} {:2.3f} Result: {}".format(id(self) % 1000, slope_diff, i_diff, e, is_parent))

        return is_parent

    def _to_dict(self):
        to_save = {
            'name': self.name,
            'observation': self._obs_id,
            'pointings': {
                'ra_dec': self._obs_info['pointings'],
                'az_el': self._obs_info['beam_az_el'].tolist()
            },
            'm': self.m,
            'intercept': self.intercept,
            'r_value': self.r_value,
            'tx': self._obs_info['transmitter_frequency'],
            'created_at': self._created_at,
            'track_size': self.size,
            'ref_data': {
                'd': self.ref_data['doppler'],
                't': self.ref_data['time'],
                'g': self.ref_data['gradient'],
                's': self.ref_data['snr'],
            },
            'data': {
                'time': self.data['time'].tolist(),
                'time_sample': self.data['time_sample'].tolist(),
                'channel': self.data['channel'].tolist(),
                'channel_sample': self.data['channel_sample'].tolist(),
                'snr': self.data['snr'].tolist(),
                'beam_id': self.data['beam_id'].tolist(),
            },
            'sampling_time': self._obs_info['sampling_time'],
            'duration': self.duration.total_seconds(),
            'activated_beams': self.activated_beams,
            'cancelled': self.cancelled,
            'terminated': self.terminated
        }

        return to_save

    def save(self):
        """
        Save the space debris candidate to disk and database

        :return:
        """

        try:
            if self._id:
                # Already saved to the database, hence we just update
                _db_model.objects.get(pk=self._id).update(**self._to_dict())
                log.info("Track {:03d} (n: {} (across {} beams), r: {:0.3f}) updated".format(id(self) % 1000, self.size,
                                                                                             self.activated_beams,
                                                                                             self.r_value))
            else:
                sd = _db_model(**self._to_dict()).save()
                self._id = sd.id
                log.info("Track {:03d} (n: {}, r: {:0.3f}) saved".format(id(self) % 1000, self.size, self.r_value))
        except ValidationError:
            log.exception("Missing or incorrect data in Space Debris Track Model")

        except OperationError:
            log.exception("Space debris track could not be saved to DB")
        else:
            self._to_save = False

    def delete(self):
        if self._id:
            try:
                _db_model.objects.get(pk=self._id).delete()
            except OperationError:
                log.exception("Track could not be deleted from DB")
            # else:
            #     log.info('Track {:03d} deleted'.format(id(self) % 1000))

    def reduce_data(self, remove_duplicate_epoch=True, remove_duplicate_channel=True):
        """

        :param remove_duplicate_epoch:
        :param remove_duplicate_channel:
        :return:
        """

        reduced_df = self.data.copy()

        if remove_duplicate_epoch:
            reduced_df = reduced_df.sort_values('snr', ascending=False).drop_duplicates(
                subset=['time_sample', 'beam_id']).sort_values(by=['time_sample'])

        if remove_duplicate_channel:
            reduced_df = reduced_df.sort_values('snr', ascending=False).drop_duplicates(
                subset=['time_sample', 'beam_id']).sort_values(by=['time_sample'])

        return reduced_df

    def aggregate_data(self, data, remove_duplicate_epoch=True, remove_duplicate_channel=True):
        """

        :param remove_duplicate_epoch:
        :param remove_duplicate_channel:
        :return:
        """

        reduced_df = data

        if remove_duplicate_epoch:
            reduced_df = data.sort_values('snr', ascending=False).drop_duplicates(
                subset=['time_sample', 'beam_id']).sort_values(by=['time_sample'])

        if remove_duplicate_channel:
            reduced_df = data.sort_values('snr', ascending=False).drop_duplicates(
                subset=['time_sample', 'beam_id']).sort_values(by=['time_sample'])

        return reduced_df

    def track_expired(self, obs_info):
        # First time sample of the track
        t_a = self.data['time'].min()

        # Last time sample of the track
        t_b = self.data['time'].max()

        # Current track time span (aka track length
        t_l = t_b - t_a

        t_l = datetime.timedelta(seconds=3)

        # Last time sample of the blob (or iteration)
        t_n = obs_info['timestamp'] + datetime.timedelta(seconds=obs_info['sampling_time'] * 160)

        # Termination threshold (twice current length (t_l): t_t = t_b +t_l)
        t_t = t_b + t_l

        return t_n > t_t

    def linear_model(self, channel, time):
        ds = self._obs_info['channel_bandwidth'] * 1e6
        dt = self._obs_info['sampling_time']

        return stats.linregress(x=time * dt, y=channel * ds)

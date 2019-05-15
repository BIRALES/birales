# -*- coding: utf-8 -*-

import datetime
import logging as log

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from mongoengine import *
from scipy import stats
from sklearn import linear_model

from pybirales import settings
from pybirales.events.events import TrackCreatedEvent, TrackModifiedEvent
from pybirales.events.publisher import EventsPublisher
from pybirales.pipeline.modules.detection.exceptions import DetectionClusterIsNotValid
from pybirales.repository.models import SpaceDebrisTrack as SpaceDebrisTrackModel

_linear_model = linear_model.RANSACRegressor(linear_model.LinearRegression())
_db_model = SpaceDebrisTrackModel


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

        # Doppler shift at

        #

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

        self._max_candidate_iter = 10

        # The iteration number
        self._iter = 0

        # Is the track saved to the database?
        self._to_save = True

        # Flag, to mark if a notification was sent
        self._notification_sent = False

        # If a beam cluster is given, add it on initialisation
        self.add(cluster)

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
        return self.data['channel'].size

    @property
    def activated_beams(self):
        return self.data['beam_id'].unique().size

    def _update(self, new_df):
        self.data = new_df
        self.m, self.intercept, self.r_value, self.p_value, self.std_err = stats.linregress(
            self.data['channel_sample'], self.data['time_sample'])

        self._last_iter_count = self.data['iter'].max()

        self.data = self.data.sort_values(by='time')

        mid = self.data['channel'].size / 2
        self.ref_data['time'] = self.data.iloc[mid]['time']
        self.ref_data['time_sample'] = self.data.iloc[mid]['time_sample']
        self.ref_data['channel'] = self.data.iloc[mid]['channel']
        self.ref_data['channel_sample'] = self.data.iloc[mid]['channel_sample']
        self.ref_data['doppler'] = (self.data.iloc[mid]['channel'] - self._obs_info['transmitter_frequency']) * 1e6
        self.ref_data['gradient'] = 1e6 * (self.data['channel'].iloc[0] - self.data['channel'].iloc[-1]) / (
                self.data['time'].iloc[0] - self.data['time'].iloc[-1]).total_seconds()
        self.ref_data['snr'] = self.data.iloc[mid]['snr']

        self._to_save = True

    def state_str(self):
        return "df:{:3.5f} Hz, df/dt:{:3.5f} Hz/s and SNR: {:2.3f} dB on {:%d.%m.%y @ %H:%M:%S} , " \
               "(C: {}, T:{}, f:{:2.5f} MHz, N:{}, B:{}, S:{:2.2f})".format(
            self.ref_data['doppler'],
            self.ref_data['gradient'],
            self.ref_data['snr'],
            self.ref_data['time'],
            self.ref_data['channel_sample'],
            self.ref_data['time_sample'],
            self.ref_data['channel'],
            self.size,
            self.activated_beams,
            self.r_value
        )

    def _fit(self, x, y):
        """
        Use a RANSAC linear fitting model to remove outliers
        todo - only apply RANSAC for clusters greater than a minimum number of data-points

        :param x:
        :param y:
        :return:
        """

        try:
            self._linear_model.fit(x.values.reshape(-1, 1), y)
        except ValueError:
            return False, self._linear_model
        else:
            return True, self._linear_model

    def add(self, cluster_df):
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

        tmp_merged_df = _merge_tmp_df(self.data, cluster_df)
        try:
            is_valid, l_model = self._fit(tmp_merged_df['channel_sample'], tmp_merged_df['time_sample'])
        except TypeError:
            print 'here'
            raise DetectionClusterIsNotValid(cluster_df)

        if is_valid:
            # Remove outliers from the merged df and update the track
            self._update(tmp_merged_df[l_model.inlier_mask_])

            # Add the SD track if the valid
            # if self.is_linear() and self.is_gradient_valid():
            #     # return False, self._linear_model
            #     return True, self._linear_model
        else:
            # If fitting failed, track data remains unchanged, and raise an exception
            raise DetectionClusterIsNotValid(cluster_df)

    def has_transitted(self, iter_count):
        """
        Determine whether this track is finished based on the current iteration
        :return:
        """

        # The track is deemed to be finished if it has not been updated since N iterations
        if (iter_count - self._last_iter_count) > self._max_candidate_iter:
            log.debug(
                'Track {:03d} (n: {}) has transitted outside detection window.'.format(id(self) % 1000, self.size))
            return True

        log.debug('Track {:03d} will be dropped in {} iterations'.format(id(self) % 1000, self._max_candidate_iter - (
                iter_count - self._last_iter_count)))

        return False

    def is_valid(self):
        """
        The candidate is not valid if size is too small or the number of activate beams is less than 2
        :return:
        """

        # Check that the number of beams activated is less than 2
        if self.data['beam_id'].unique().size < 2:
            return False

        # Check that the candidate has the minimum number of unique channel and time data points
        if self.data['channel'].unique().size < 5 or self.data['time'].unique().size < 5:
            return False

        if not self.is_linear():
            return False

        if not self.is_gradient_valid():
            return False

        return True

    def is_linear(self):
        return np.abs(self.r_value) >= settings.detection.linearity_thold

    def is_gradient_valid(self):
        # print settings.detection.gradient_thold[0], self.ref_data['gradient'], settings.detection.gradient_thold[1],
        # print settings.detection.gradient_thold[0] >= self.ref_data['gradient'] >= settings.detection.gradient_thold[1]
        return settings.detection.gradient_thold[0] >= self.ref_data['gradient'] >= settings.detection.gradient_thold[
            1]

    def is_parent_of(self, detection_cluster):
        """
        Determine whether a detection cluster should be associated with this space debris track by giving
        a similarity score. Similarity between beam candidate and space debris track is determined by
        cosine distance.

        :param detection_cluster: The detection cluster with which the track is being compared
        :return:
        """

        cluster_m, cluster_c, _, _, _ = stats.linregress(detection_cluster['channel_sample'],
                                                         detection_cluster['time_sample'])

        return dist.cosine([self.m, self.intercept], [cluster_m, cluster_c]) < settings.detection.similarity_thold

    def _to_dict(self):
        return {
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
            'created_at': datetime.datetime.utcnow(),
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
            'activated_beams': self.activated_beams
        }

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
                # print self._to_dict()
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
            else:
                log.info('Track {:03d} deleted'.format(id(self) % 1000))

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

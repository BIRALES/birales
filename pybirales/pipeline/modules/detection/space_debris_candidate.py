import datetime
import logging as log

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from mongoengine import *
from scipy import stats
from sklearn import linear_model

from pybirales import settings
from pybirales.pipeline.modules.detection.exceptions import DetectionClusterIsNotValid
from pybirales.repository.models import SpaceDebrisTrack

_linear_model = linear_model.RANSACRegressor(linear_model.LinearRegression())
_db_model = SpaceDebrisTrack


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
        self._id = None
        self.name = None
        self._created_at = datetime.datetime.utcnow()
        self._obs_info = obs_info
        self._obs_id = settings.observation.id

        self._linear_model = _linear_model
        # The attributes of the linear model
        self.m = None
        self.intercept = None
        self.r_value = None

        # The radar cross section of the track
        self._rcs = None

        # The target we are trying to track
        self._target = None

        # The target which most closely matches the parameters
        self._detected_target = None

        # DataFrame encapsulating the track data (snr, time, channel, beam id)
        self.data = pd.DataFrame(columns=['time_sample', 'channel_sample', 'time', 'channel', 'snr', 'beam_id'])

        # The time at which the track is expected to exit the detection window
        self._exit_time = None

        # The last iteration counter
        self._last_iter_count = 0

        self._max_candidate_iter = 10

        # The iteration number
        self._iter = 0

        # If a beam cluster is given, add it on initialisation
        self.add(cluster)

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

        # Save the candidate
        self._save()

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
        is_valid, l_model = self._fit(tmp_merged_df['channel_sample'], tmp_merged_df['time_sample'])

        if is_valid:
            # Remove outliers from the merged df and update the track
            self._update(tmp_merged_df[l_model.inlier_mask_])
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
            log.debug('Track {} (n: {}) has transitted outside detection window.'.format(id(self), self.size))
            return True

        log.debug('Track {} will be dropped in {} iterations'.format(id(self), self._max_candidate_iter - (
                iter_count - self._last_iter_count)))

        return False


    def is_valid(self):
        if self.size < 5 or self.activated_beams < 2:
            return False

        return True

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

    def _save(self):
        """
        Save the space debris candidate to disk and database

        :return:
        """

        # Persist the space debris detection to the database
        if settings.detection.save_candidates:
            try:
                self._save_db()
                log.info("Space debris track {} saved with {} data points".format(id(self), self.size))
            except ValidationError:
                log.exception("Missing or incorrect data in Space Debris Track Model")
            except OperationError:
                log.exception("Space debris track could not be saved to DB")

                # Upload the space debris detection to an FTP server

    def _save_db(self):
        """
        Return a dict representation of the SpaceDebrisTrack

        :return:
        """

        def _data(df):
            return {
                'time': df['time'].tolist(),
                'time_sample': df['time_sample'].tolist(),
                'channel': df['channel'].tolist(),
                'channel_sample': df['channel_sample'].tolist(),
                'snr': df['snr'].tolist(),
                'beam_id': df['beam_id'].tolist(),
            }

        if self._id:
            # Already saved to the database, hence we just update
            _db_model.objects.get(pk=self._id).update(data=_data(self.data),
                                                      duration=self.duration.total_seconds(),
                                                      m=self.m, r_value=self.r_value, intercept=self.intercept,
                                                      track_size=self.size)
        else:
            sd = _db_model(**{
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
                'data': _data(self.data),
                'sampling_time': self._obs_info['sampling_time'],
                'duration': self.duration.total_seconds()
            }).save()

            self._id = sd.id

    def delete(self):
        if self._id:
            # Already saved to the database, hence we just update
            _db_model.objects.get(pk=self._id).delete()

            log.info('Track {} deleted'.format(id(self)))

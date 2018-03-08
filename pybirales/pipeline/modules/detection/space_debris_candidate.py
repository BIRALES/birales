import datetime
import logging as log

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from mongoengine import *
from sklearn import linear_model

from pybirales import settings
from pybirales.pipeline.modules.detection.detection_clusters import DetectionCluster
from pybirales.pipeline.modules.detection.tdm.tdm import TDMWriter, CSVWriter
from pybirales.repository.models import SpaceDebrisTrack

_linear_model = linear_model.RANSACRegressor(linear_model.LinearRegression())
_db_model = SpaceDebrisTrack
_tmd_writer = TDMWriter()
_csv_writer = CSVWriter()


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
    def __init__(self, obs_info, beam_candidate=None):
        """

        :param obs_info:
        :param beam_candidate:
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
        self.score = None

        # The noise value per beam
        self._beam_noise = np.zeros(settings.beamformer.nbeams)

        # The radar cross section of the track
        self._rcs = None

        # The target we are trying to track
        self._target = None

        # The target which most closely matches the parameters
        self._detected_target = None

        # DataFrame encapsulating the track data (snr, time, channel, beam id)
        self.data = pd.DataFrame(columns=['time_sample', 'time', 'channel', 'snr', 'beam_id'])
        # self.data.set_index('time_sample')

        # The filepath where to save the space debris candidate
        self._tdm_filepath = None

        # The filepath where to save the space debris candidate (for debug purposes)
        self._debug_filepath = None

        # The time at which the track is expected to exit the detection window
        self._exit_time = None

        # If a beam candidate is given, add it on initialisation
        if isinstance(beam_candidate, DetectionCluster):
            self.add(beam_candidate)

    @property
    def duration(self):
        return self.data['time'].max() - self.data['time'].min()

    def add(self, beam_candidate):
        """
        Associate a beam candidate with this space debris track.
        Update the space debris track's attributes based on new data

        :param beam_candidate:
        :type beam_candidate: DetectionCluster
        :return:
        """
        if not self.is_linear(beam_candidate):
            log.warning("Won't add this beam candidate")
            return

        temp_df = pd.DataFrame({
            'time_sample': beam_candidate.time_data,
            'time': beam_candidate.time_f_data,
            'channel': beam_candidate.channel_data,
            'snr': beam_candidate.snr_data,
            'beam_id': [beam_candidate.beam_id for _ in range(0, len(beam_candidate.time_data))],
        })

        if not self.data.empty:
            # Combine the beam candidate track to this track
            self.data = pd.concat([self.data, temp_df])
        else:
            self.data = temp_df

            # Record the noise of the beam
        self._beam_noise[beam_candidate.beam_id] = beam_candidate.beam_config['beam_noise']

        # Update linear model of the track
        channels, time = self.data['channel'].values.reshape(-1, 1), self.data['time_sample']

        try:
            self._linear_model.fit(channels, time)
        except ValueError:
            log.debug('Space debris linear fitting failed. Beam candidate not added to track {}'.format(id(self)))
        else:
            channels, time = channels[self._linear_model.inlier_mask_], time[self._linear_model.inlier_mask_]

            # todo - replace the functionality below with an _update function
            self.score = self._linear_model.estimator_.score(channels, time)
            self.m = self._linear_model.estimator_.coef_[0]
            self.intercept = self._linear_model.estimator_.intercept_

            # Remove outliers from the data
            self.data = self.data[self._linear_model.inlier_mask_]

            # Sort data by time
            self.data = self.data.sort_values(by='time')

            # Update rcs
            # Update detected target

            # Persist the candidate to DB/TDM
            self._save()

    def is_linear(self, detection_cluster):
        """
        Check that if the data from the detection cluster is added to the space debris track,
        the linear fit works

        todo - on success, you should avoid doing this twice (when adding)

        :param detection_cluster:
        :return:
        """

        temp_df = pd.DataFrame({
            'time_sample': detection_cluster.time_data,
            'channel': detection_cluster.channel_data,
            'snr': detection_cluster.snr_data,
            'beam_id': [detection_cluster.beam_id for _ in range(0, len(detection_cluster.time_data))],
        })

        temp_df2 = temp_df
        if not self.data.empty:
            # Combine the beam candidate track to this track
            temp_df2 = pd.concat([self.data, temp_df])

        channels, time = temp_df2['channel'].values.reshape(-1, 1), temp_df2['time_sample']

        try:
            self._linear_model.fit(channels, time)
        except ValueError:
            log.warning("Linear Fit failed when combining beam candidate {} to the space debris track {}".format(
                id(detection_cluster), id(self)
            ))

            return False
        else:
            channels, time = channels[self._linear_model.inlier_mask_], time[self._linear_model.inlier_mask_]

            score = self._linear_model.estimator_.score(channels, time)

            return score > 0.9 and settings.detection.m_limit[0] <= self._linear_model.estimator_.coef_[0] <= \
                                   settings.detection.m_limit[1]

    def is_finished(self, current_time, min_channel):
        """
        Determine whether this track is finished based on the current time
        time_max = ( channel_min - intercept ) / gradient
        :return:
        """
        self._exit_time = (min_channel - self.intercept) / self.m

        return current_time > self._exit_time

    def is_parent_of(self, detection_cluster, similarity_thold=0.1):
        """
        Determine whether a detection cluster should be associated with this space debris track by giving
        a similarity score. Similarity between beam candidate and space debris track is determined by
        cosine distance.

        :param detection_cluster: The detection cluster with which the track is being compared
        :param similarity_thold: The similarity threshold which determines if track is similar or not
        :type detection_cluster: DetectionCluster
        :type similarity_thold: float
        :return:
        """

        if self.m and self.intercept:
            return dist.cosine([self.m, self.intercept], [detection_cluster.m, detection_cluster.c]) < similarity_thold

        return True

    def _save(self):
        """
        Save the space debris candidate to disk and database

        :return:
        """

        # Persist the space debris detection as a TDM file
        if settings.detection.save_tdm:
            self._tdm_filepath = _tmd_writer.write(self._tdm_filepath, self._obs_info, self)

        # Save the detection cluster as CSV file for analysis
        if settings.detection.debug_candidates:
            _csv_writer.write(self._debug_filepath, self)

        # Persist the space debris detection to the database
        if settings.detection.save_candidates:
            try:
                self._save_db()
                log.info("Space debris track saved")
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
                'snr': df['snr'].tolist(),
                'beam_id': df['beam_id'].tolist(),
            }

        if self._id:
            # Already saved to the database, hence we just update
            _db_model.objects.get(pk=self._id).update(data=_data(self.data),
                                                      beam_noise=self._beam_noise.tolist(),
                                                      duration=self.duration.total_seconds())
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
                'score': self.score,
                'tx': self._obs_info['transmitter_frequency'],
                'created_at': datetime.datetime.utcnow(),
                'beam_noise': self._beam_noise.tolist(),
                'size': self.data.shape[1],
                'tdm_filepath': self._tdm_filepath,
                'data': _data(self.data),
                'duration': self.duration.total_seconds()
            }).save()

            self._id = sd.id

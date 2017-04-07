import numpy as np
import datetime

from astropy.time import Time
from pybirales.base import settings
import logging as log


class DetectionCluster:
    def __init__(self, beam_id, time, channels, snr):
        """
        Initialisation of the Detection cluster Object

        :param beam_id:
        :param time:
        :param channels:
        :param snr:
        """

        self.beam_id = beam_id
        self._processed_time = datetime.datetime.utcnow()
        self.id = str(self.beam_id) + '.' + self._processed_time.isoformat()
        self.to_delete = False
        self.to_save = True

        self.time_data = time
        self.channel_data = channels
        self.snr_data = snr

        self.m = 0.0
        self.c = 0.0
        self._score = 0.0

        self.illumination_time = np.min(self.time_data)

    def is_linear(self, model, threshold):
        """
        Determine if cluster is a linear cluster (Shaped as a line)

        :param model: The model which will determine if cluster is linear as expected
        :param threshold:
        :type threshold: float
        :return:
        """

        try:
            channels = [[channel] for channel in self.channel_data]
            model.fit(channels, self.time_data)
        except ValueError:
            log.debug('Linear interpolation failed. No inliers found.')
        else:
            # todo - check what you are going to do about the inliers mask
            # Create mask of inlier data points
            # inlier_mask = self._model.inlier_mask_

            # Remove outliers - select data points that are inliers
            # self.data = cluster_data[inlier_mask]

            self._score = model.estimator_.score(channels, self.time_data)
            self.m = model.estimator_.coef_[0]
            self.c = model.estimator_.intercept_

            if self._score < threshold:
                return False
            return True

    def is_similar_to(self, cluster, threshold):
        """
        Determine if two clusters are similar

        :param cluster: The cluster we are comparing to
        :param threshold:
        :type cluster: DetectionCluster
        :type threshold: float
        :return:
        """

        # The gradients of the clusters are similar
        if self._percentage_difference(cluster.m, self.m) <= threshold:
            # The intercept of the clusters are similar
            if self._percentage_difference(cluster.c, self.c) <= threshold:
                return True

        return False

    @staticmethod
    def _percentage_difference(a, b):
        """
        Calculate the difference between two values
        :param a:
        :param b:
        :return:
        """
        diff = a - b
        mean = np.mean([a, b])
        try:
            percentage_difference = abs(diff / mean)
        except RuntimeWarning:
            percentage_difference = 1.0

        return percentage_difference

    def merge(self, cluster):
        # Return a new Detection Cluster with the merged data
        return DetectionCluster(beam_id=cluster.beam_id,
                                time=np.concatenate([self.time_data, cluster.time_data]),
                                channels=np.concatenate([self.channel_data, cluster.channel_data]),
                                snr=np.concatenate([self.snr_data, cluster.snr_data]))

    def delete(self):
        """
        Mark this detection cluster for deletion
        :return: void
        """

        self.to_delete = True

    def saved(self):
        """
        Mark this cluster as already saved to the database
        :return: void
        """

        self.to_save = False

    def get_detections(self):
        """
        Returns the detection data of the beam cluster in a list that can be
        easily converted to json

        :return:
        """
        return [
            {
                'time': self._timestamp(elapsed_time),
                'mdj2000': self._get_mjd2000(elapsed_time),
                'time_elapsed': self._time_elapsed(elapsed_time),
                'frequency': frequency,
                'doppler_shift': self._get_doppler_shift(settings.observation.transmitter_frequency, frequency),
                'snr': float(snr),
            } for frequency, elapsed_time, snr in zip(self.channel_data, self.time_data, self.snr_data)
        ]

    @staticmethod
    def _get_doppler_shift(transmission_frequency, reflected_frequency):
        return (reflected_frequency - transmission_frequency) * 1e6

    @staticmethod
    def _time_elapsed(elapsed_time):
        return elapsed_time

    @staticmethod
    def _time(time):
        # ref_time = self.beam.data_set.config['timestamp'] / 1000.
        return Time(time, format='unix')

    def _timestamp(self, elapsed_time):
        time = self._time(elapsed_time)
        return time.iso

    def _get_mjd2000(self, elapsed_time):
        time = self._time(elapsed_time)
        return time.mjd

    def to_json(self):
        return {
            '_id': self.id,
            'beam_id': self.beam_id,
            'tx': settings.observation.transmitter_frequency,
            'illumination_time': self.illumination_time,
            'created_at': datetime.datetime.utcnow(),
            'data': {
                'time': self.time_data.tolist(),
                'channel': self.channel_data.tolist(),
                'snr': self.snr_data.tolist(),
            }
        }

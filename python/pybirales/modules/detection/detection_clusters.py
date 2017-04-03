import numpy as np
import datetime

from sklearn import linear_model
from astropy.time import Time
from pybirales.base import settings


class DetectionCluster:
    m = None
    c = None
    data = None

    _score = None
    _model = None

    def __init__(self, beam_id, cluster_data):
        """
        Initialisation of the Detection cluster Object

        :param cluster_data:
        :type cluster_data: numpy.dtype
        """

        self._model = linear_model.RANSACRegressor(linear_model.LinearRegression())
        x = cluster_data[:, [1]]
        y = cluster_data[:, 0]

        self._model.fit(x, y)

        # Create mask of inlier data points
        inlier_mask = self._model.inlier_mask_

        # Remove outliers - select data points that are inliers
        self.data = cluster_data[inlier_mask]

        # Set public properties of cluster
        self._score = self._model.estimator_.score(x, y)
        self.m = self._model.estimator_.coef_[0]
        self.c = self._model.estimator_.intercept_

        # ---------------
        self.beam_id = beam_id
        self._processed_time = datetime.datetime.utcnow()
        self.id = str(self.beam_id) + '.' + self._processed_time.isoformat()
        self.detections = []
        self.illumination_time = np.min(x)  # get minimum time

        self.to_delete = False
        self.to_save = True

        # todo - This function can be called lazily, since it is only for visualisation
        self.set_data(self.data)

    def is_linear(self, threshold):
        """
        Determine if cluster is a linear cluster (Shaped as a line)

        :param threshold:
        :type threshold: float
        :return:
        """

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
        merged_data = np.concatenate((self.data, cluster.data))

        # Return a new Detection Cluster with the merged data
        return DetectionCluster(merged_data)

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

    # --------------
    def set_data(self, detection_data):
        for frequency, elapsed_time, snr in detection_data:
            self.detections.append({
                'time': self._timestamp(elapsed_time),
                'mdj2000': self._get_mjd2000(elapsed_time),
                'time_elapsed': self._time_elapsed(elapsed_time),
                'frequency': frequency,
                'doppler_shift': self._get_doppler_shift(settings.observation.transmitter_frequency, frequency),
                'snr': snr,
            })

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
                'snr': snr,
            } for frequency, elapsed_time, snr in self.data
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

    def __iter__(self):
        yield '_id', self.id
        yield 'name', self.name
        yield 'detections', self.get_detections()
        yield 'beam_id', self.beam_id
        # yield 'data_set_id', self.beam.data_set.id
        yield 'illumination_time', self.illumination_time
        yield 'created_at', datetime.datetime.now().isoformat()

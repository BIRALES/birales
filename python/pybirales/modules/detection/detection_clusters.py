import numpy as np
import datetime
import logging as log
import time as time2

from pybirales.base import settings

np.set_printoptions(precision=20)


class DetectionCluster:
    def __init__(self, model, beam_config, time_data, channels, snr):
        """
        Initialisation of the Detection cluster Object

        :param model: The model which will determine if cluster is linear as expected
        :param beam:
        :param time_data:
        :param channels:
        :param snr:

        :type time_data: numpy.array of Time
        :type beam: Beam
        :type channels: numpy.array of floats
        :type snr: numpy.array of floats
        """

        self._model = model
        self.beam_config = beam_config

        self._processed_time = datetime.datetime.utcnow()
        self.id = str(self.beam_config['beam_id']) + '_' + self._processed_time.isoformat()
        self.to_delete = False
        self.to_save = True

        self.time_data = time_data
        self.channel_data = channels
        self.snr_data = snr

        self.min_time = np.min(self.time_data).astype('M8[ms]').astype('O')
        self.max_time = np.max(self.time_data).astype('M8[ms]').astype('O')
        self.min_channel = np.min(self.channel_data)
        self.max_channel = np.max(self.channel_data)

        self.m = np.nan
        self.c = np.nan
        self.score = np.nan

        # Compare the detection cluster's data against a (linear) model
        t = time2.time()
        self.fit_model(model=self._model, channel_data=self.channel_data, time_data=self.time_data)
        log.debug('Fitting on %s data points took %0.3f s', len(self.channel_data), time2.time() - t)

    def fit_model(self, model, channel_data, time_data):
        """
        Compare the detections cluster data against a model
        todo - this function has to be refactored
        :return:
        """

        if settings.detection.select_highest_snr:
            c = np.array(channel_data)
            ts = np.array(time_data)
            if np.all(c == c[0]) or np.all(ts == ts[0]):
                self.score = np.nan
                self.m = np.nan
                self.c = np.nan
                return

            s = np.array(self.snr_data)

            ndx = np.lexsort(keys=(s, ts))
            index = np.empty(len(ts), 'bool')
            index[-1] = True
            index[:-1] = ts[1:] != ts[:-1]
            i = ndx[index]

            # channels = np.array([[channel, ss] for channel, ss in zip(c[i], s[i])])
            channels = np.array([[channel] for channel in c[i]])
            time = ts[i].astype('int64') * 1e-9
        else:
            channels = np.array([[channel] for channel in channel_data])
            time = np.array(time_data).astype('int64') * 1e-9

        try:
            t = time2.time()
            model.fit(channels, time)

            t2 = time2.time() - t
            log.debug('Fitting 2 took %0.3f s', t2)

        except ValueError:
            log.debug('Linear interpolation failed. No inliers found.')
        else:
            # todo - check what you are going to do about the inliers mask
            # Create mask of inlier data points
            inlier_mask = model.inlier_mask_

            # Remove outliers - select data points that are inliers
            channels = channels[inlier_mask]
            time = time[inlier_mask]
            if settings.detection.select_highest_snr:
                snr = self.snr_data[i][inlier_mask]
            else:
                snr = self.snr_data[inlier_mask]

            self.score = model.estimator_.score(channels, time)
            self.m = model.estimator_.coef_[0]
            self.c = model.estimator_.intercept_

            self.channel_data = np.array([channel[0] for channel in channels])
            # self.time_data = [Time(t, format='unix') for t in time]
            self.time_data = [np.datetime64(int(t*1e9), 'ns') for t in time]
            self.snr_data = np.array(snr)

    def is_linear(self, threshold):
        """
        Determine if cluster is a linear cluster up to a certain threshold

        :param threshold:
        :type threshold: float

        :return:
        """

        return self.score > threshold

    def is_valid(self):
        if len(self.time_data) > 2:
            if settings.detection.m_limit[0] <= self.m <= settings.detection.m_limit[1]:
                return True

        return False

    def is_similar_to(self, cluster, threshold):
        """
        Determine if two clusters are similar

        :param cluster: The cluster we are comparing to
        :param threshold:
        :type cluster: DetectionCluster
        :type threshold: float
        :return:
        """

        temp = DetectionCluster(model=self._model,
                                beam_config=cluster.beam_config,
                                time_data=np.concatenate([self.time_data, cluster.time_data]),
                                channels=np.concatenate([self.channel_data, cluster.channel_data]),
                                snr=np.concatenate([self.snr_data, cluster.snr_data]))

        merge = self._pd(cluster.m, self.m) <= threshold and self._pd(cluster.c, self.c) <= threshold

        if temp.score >= self.score or merge:
            return True

        return False

    @staticmethod
    def _pd(a, b):
        """
        Calculate the percentage difference between two values
        :param a:
        :param b:
        :return:
        """
        diff = a - b
        mean = np.mean([a, b])
        try:
            percentage_difference = abs(diff / mean)

            if np.isnan(percentage_difference):
                percentage_difference = 1.0
        except RuntimeWarning:
            percentage_difference = 1.0

        return percentage_difference

    def merge(self, cluster):
        """
        Create a new Cluster from the (merged) data of this cluster and that of a new cluster.

        :param cluster:
        :type cluster: DetectionCluster
        :return: DetectionCluster
        """

        return DetectionCluster(model=self._model,
                                beam_config=cluster.beam_config,
                                time_data=np.concatenate([self.time_data, cluster.time_data]),
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

    @staticmethod
    def _get_doppler_shift(transmission_frequency, reflected_frequency):
        return (reflected_frequency - transmission_frequency) * 1e6

    @staticmethod
    def _time_elapsed(elapsed_time):
        return elapsed_time

    @staticmethod
    def _time(time):
        # ref_time = self.beam.data_set.config['timestamp'] / 1000.

        return time
        # return Time(time, format='unix')

    def _timestamp(self, elapsed_time):
        return self._time(elapsed_time).iso

    def _get_mjd2000(self, elapsed_time):
        return self._time(elapsed_time).mjd

    def to_json(self):
        max_snr_mask = np.argmax(self.snr_data)
        return {
            '_id': self.id,
            'beam': {
                'id': self.beam_config['beam_id'],
                'ra': self.beam_config['beam_ra'],
                'dec': self.beam_config['beam_dec'],
            },
            'model': {
                'm': self.m,
                'c': self.c,
                'score': self.score,
            },
            'beam_id': self.beam_config['beam_id'],
            'tx': settings.observation.transmitter_frequency,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'min_channel': self.min_channel,
            'max_channel': self.max_channel,
            'created_at': datetime.datetime.utcnow(),
            'configuration_id': self.beam_config['configuration_id'],
            'noise': self.beam_config['beam_noise'],
            'size': len(self.channel_data),
            'data': {
                'time': [np.datetime_as_string(b) for b in self.time_data],
                'channel': self.channel_data.tolist(),
                'snr': self.snr_data.tolist(),
            },
            'max_snr': {
                'time': float(self.time_data[max_snr_mask]),
                'channel': float(self.channel_data[max_snr_mask]),
                'snr': float(self.snr_data[max_snr_mask]),
            }
        }

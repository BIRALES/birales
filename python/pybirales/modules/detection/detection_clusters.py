import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
import logging as log


class DetectionCluster:
    m = None
    c = None
    data = None

    _score = None
    _model = None

    def __init__(self, cluster_data):
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

    def is_cluster_similar(self, cluster, threshold):
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

    def _visualise_cluster(self):
        if config.get_boolean('debug', 'DEBUG_CANDIDATES'):
            # cluster_equation = 'm=' + cluster['m'] + ', c=' + cluster['c'] + ', r=' + cluster['r']
            plt.plot(self.data, 'o', label='')
            plt.legend(loc='best', fancybox=True, framealpha=0.9)
            plt.xlabel('Channel')
            plt.title('Detection Cluster')
            plt.ylabel('Time')
            plt.show()
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
from sklearn.cluster import DBSCAN

from pybirales import settings
from pybirales.pipeline.base.timing import timeit
from pybirales.pipeline.modules.detection.exceptions import NoDetectionClustersFound

_eps = 5
_min_samples = 5
_algorithm = 'kd_tree'
_linear_model = linear_model.RANSACRegressor(linear_model.LinearRegression())
db_scan = DBSCAN(eps=_eps, min_samples=_min_samples, algorithm=_algorithm, n_jobs=-1)
N_SAMPLES = 32
N_SAMPLES = 160


def _validate(channel, time_sample, td):
    """
    Validate the detection cluster

    :param channel_ndx:
    :param time_ndx:
    :return:

    """

    # Check if cluster is not small
    if channel.shape[0] < 3:
        # log.debug('Rejecting cluster with n: %s', channel_ndx.shape[0])
        return False

    m, _, r_value, _, _ = stats.linregress(time_sample, channel)

    # Correct the gradient's units to be in Hz/s
    m = m * 1e6 / (td / np.timedelta64(1, 's'))

    # Check if cluster is linear
    if np.abs(r_value) < settings.detection.linearity_thold:
        # log.debug('Rejecting cluster with r: %s', r_value)
        return False

    # Check if cluster gradient is within doppler gradient window
    if settings.detection.enable_gradient_thold:
        if not (settings.detection.gradient_thold[0] >= m >= settings.detection.gradient_thold[1]):
            # log.debug('Rejecting cluster with m: %s', m)
            return False

    return True


def _create(snr_data, cluster_data, channels, t0, td, beam_id, iter_count, channel_noise):
    """

    :param cluster_data: The labelled cluster data
    :param channels:
    :param t0:
    :param td:
    :param beam_id:
    :param iter_count:
    :return:
    """

    channel_ndx = cluster_data[:, 0]
    time_ndx = cluster_data[:, 1]
    time_sample = time_ndx + iter_count * N_SAMPLES

    # Calculate the channel (frequency) of the sample from the channel index
    channel = channels[channel_ndx]

    # Check if the cluster is a valid cluster
    if not _validate(channel, time_sample, td):
        return False, None

    # Calculate the time of the sample from the time index
    time = t0 + (time_sample - 160 * iter_count) * td

    cluster_data = np.append(cluster_data, np.expand_dims(np.full(len(cluster_data), beam_id), axis=1), axis=1)

    # return True, create_candidate(cluster_data, channels, iter_count, 160, t0, td, channel_noise, beam_id)

    return True, pd.DataFrame({
        'time_sample': time_sample,
        'channel_sample': channel_ndx,
        'time': time,
        'channel': channel,
        'snr': 10 * np.log10(snr_data / np.mean(channel_noise[beam_id])),
        'beam_id': np.full(time_ndx.shape[0], beam_id),
        'iter': np.full(time_ndx.shape[0], iter_count),
    })


def partition_input_data(input_data, channel_noise, beam_id):
    """

    :param input_data:
    :param channel_noise:
    :param beam_id:
    :return:
    """

    # Process part of the input_data
    beam_data = input_data[beam_id, :, :]

    # Get the channel noise for this beam
    beam_noise = channel_noise[beam_id]

    """
    # Calculate the SNR
    snr = beam_data - beam_noise[:, np.newaxis]

    # Ignore values where SNR < 0. Ie. where signal power is < than background noise
    ndx = np.where(snr > 0.)

    # Transform them in a time (x), channel (y) nd-array and snr
    return np.column_stack(ndx), snr[ndx]
    
    """

    # beam_data = beam_data - np.mean(beam_noise)
    ndx = np.where(beam_data > 0.)
    power = beam_data[ndx]
    return np.column_stack(ndx), power

    # Remove filtered data from the calculation
    noise_estimate = np.mean(beam_noise)

    # snr_data = (beam_data - noise_estimate) / noise_estimate

    # Noise estimate already removed in pre-processor module
    snr_data = beam_data / noise_estimate

    # Remove filtered data from the calculation
    ndx = np.where(snr_data > 0.)
    snr_data = snr_data[ndx]

    # Convert SNR to dB
    snr_data = 10 * np.log10(snr_data)

    return np.column_stack(ndx), snr_data


def dbscan_clustering(beam_ndx, snr_data):
    """

    :param beam_ndx:
    :param snr_data:
    :return:
    """
    try:
        # Perform (2D) clustering on the data and returns cluster labels the points are associated with
        c_labels = db_scan.fit_predict(beam_ndx)
    except ValueError:
        raise NoDetectionClustersFound

    if len(c_labels) < 1:
        raise NoDetectionClustersFound

    # Add cluster labels to the data
    labelled_data = np.append(beam_ndx, np.expand_dims(c_labels, axis=1), axis=1)

    # Cluster mask to remove noise clusters
    denoise_mask = labelled_data[:, 2] > -1

    # if len(snr_data[denoise_mask]) != 0:
    #     print(snr_data[denoise_mask])
    # Select only those labels which were not classified as noise was(-1)
    return labelled_data[denoise_mask], snr_data[denoise_mask]


def create_clusters(snr_data, labelled_data, channels, t0, td, beam_id, iter_count, channel_noise):
    """
    Group the data points in clusters
    :param snr_data:
    :param labelled_data:
    :param channels:
    :param t0:
    :param td:
    :param beam_id:
    :param iter_count:
    :return:
    """

    clusters = []

    unique_cluster_labels = np.unique(labelled_data[:, 2]).tolist()

    for label in unique_cluster_labels:
        label_mask = labelled_data[:, 2] == label
        valid, cluster = _create(snr_data[label_mask], labelled_data[label_mask], channels, t0, td, beam_id,
                                 iter_count, channel_noise)

        if valid:
            clusters.append(cluster)

    return clusters


@timeit
def detect(input_data, channels, t0, td, iter_count, channel_noise, beam_id):
    """
    Use the DBScan algorithm to create a set of clusters from the given beam data

    To be run in parallel using the multi-processing queue

    :param input_data:
    :param beam_id:
    :param channels:
    :param t0:
    :param td:
    :param iter_count:
    :param channel_noise:
    :return:
    """

    # Process a slice of the input data
    beam_ndx, snr_data = partition_input_data(input_data, channel_noise, beam_id)

    try:
        # Run the clustering algorithm on the beam data (channel,time,snr,label)
        labelled_data, snr_data = dbscan_clustering(beam_ndx, snr_data)
    except NoDetectionClustersFound:
        # log.debug('No detection clusters were found in iteration {} (beam {})'.format(iter_count, beam_id))
        return []

    # Validate and create linear detection clusters
    clusters = create_clusters(snr_data, labelled_data, channels, t0, td, beam_id, iter_count, channel_noise)

    return clusters

import logging as log
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from astropy.io import fits
from pybirales import settings
from pybirales.pipeline.modules.detection.detection_clusters import DetectionCluster
from pybirales.pipeline.base.timing import timeit
from sklearn import linear_model
from sklearn.cluster import DBSCAN

_eps = 5
_min_samples = 5
_algorithm = 'kd_tree'
_linear_model = linear_model.RANSACRegressor(linear_model.LinearRegression())
db_scan = DBSCAN(eps=_eps, min_samples=_min_samples, algorithm=_algorithm, n_jobs=-1)
_ref_time = None
_time_delta = None


def _db_scan(beam_id, data):
    """
    Perform the db_scan algorithm on the beam data to identify clusters

    :param beam_id:
    :param data:
    :return:
    """

    try:
        # Perform clustering on the data and returns cluster labels the points are associated with
        return db_scan.fit_predict(data)
    except ValueError:
        return []


def check_doppler_range(channels, tx):
    # If Doppler filtering is disabled, all candidates are valid
    if not settings.detection.doppler_subset:
        return True

    dp = (channels - tx) * 1e6
    return (dp > settings.detection.doppler_range[0]).all() and (dp < settings.detection.doppler_range[1]).all()


def _create_detection_clusters(data, beam, cluster_labels, label):
    data_indices = data[np.where(cluster_labels == label)]

    log.debug('Beam %s: cluster %s contains %s data points', beam.id, label, len(data_indices[:, 1]))

    channel_indices = data_indices[:, 1]
    time_indices = data_indices[:, 0]

    # If cluster is 'small' do not consider it
    if len(channel_indices) < _min_samples:
        log.debug('Beam %s: Ignoring small cluster with %s data points', beam.id, len(channel_indices))
        return None

    # If cluster is not withing physical doppler range limits
    if not check_doppler_range(beam.channels[channel_indices], beam.tx):
        return None

    # Create a Detection Cluster from the cluster data
    cluster = DetectionCluster(model=_linear_model,
                               beam_config=beam.get_config(),
                               time_data=[_ref_time + t * _time_delta for t in beam.time[time_indices]],
                               channels=beam.channels[channel_indices],
                               snr=beam.snr[(time_indices, channel_indices)])

    # Add only those clusters that are linear
    if cluster.is_linear(threshold=0.9) and cluster.is_valid():
        log.debug('Beam %s: Cluster with m:%3.2f, c:%3.2f, n:%s and r:%0.2f is considered to be linear.', beam.id,
                  cluster.m,
                  cluster.c, len(channel_indices), cluster.score)
        return cluster


@timeit
def _detect_clusters(beam):
    """
    Use the DBScan algorithm to create a set of clusters from the given beam data

    :param beam: The beam object from which the clusters will be generated
    :return:
    """

    # Select the data points that are non-zero
    ndx = np.where(beam.snr > 0)

    # Transform them in a time (x), channel (y) nd-array
    data = np.column_stack(ndx)

    # Perform DBScan on the cluster data
    c_labels = _db_scan(beam.id, data)

    log.debug('Beam %s: DBSCAN identified %s cluster labels', beam.id, len(c_labels))

    if len(c_labels) < 1:
        return []

    # Select only those labels which were not classified as noise (-1)
    filtered_c_labels = c_labels[c_labels > -1]

    log.debug('Beam %s: %s / %s data points was considered to be noise',
              beam.id,
              len(c_labels) - len(filtered_c_labels),
              len(c_labels))

    log.debug('Beam %s: %s unique clusters were detected', beam.id, len(np.unique(filtered_c_labels)))

    # Group the data points in clusters
    clusters = []
    append = clusters.append
    unique_c_labels = np.unique(filtered_c_labels).tolist()
    for label in unique_c_labels:
        cluster = _create_detection_clusters(data, beam, c_labels, label)
        if cluster:
            append(cluster)

    log.debug('Beam %s: DBSCAN detected %s (from %s) valid clusters', beam.id, len(clusters), len(unique_c_labels))

    return clusters


@timeit
def save_fits(data, data_type, beam_id):
    if beam_id in settings.detection.visualise_beams:
        fits_filename = os.path.join(os.environ['HOME'], settings.detection.visualise_fits_dir,
                                     settings.observation.name,
                                     '{}_{}.fits'.format(data_type, beam_id))
        try:
            fits_file = fits.open(fits_filename)
            new_data = np.vstack([fits_file[0].data, data])
            fits.writeto(fits_filename, new_data, overwrite=True)
        except IOError:
            fits.writeto(fits_filename, data, overwrite=True)


@timeit
def detect(obs_info, queue, beam):
    """
    The core detection algorithm to be applied to the incoming data

    To be run in parallel using the multi-processing queue

    :param obs_info:
    :param queue:
    :param beam:
    :return:
    """

    global _time_delta, _ref_time
    _ref_time = np.datetime64(obs_info['timestamp'])
    _time_delta = np.timedelta64(int(obs_info['sampling_time'] * 1e9), 'ns')

    # Associate a beam candidate repository to the queue
    # queue.set_repository(BeamCandidateRepository())

    # Save the raw data as a fits file
    save_fits(beam.snr, 'raw', beam.id)

    # Apply the pre-processing filters to the beam data
    beam.apply_filters()

    # Save the filtered data as a fits file
    save_fits(beam.snr, 'filtered', beam.id)

    # Add the beam candidates to the beam queue
    enqueue = queue.enqueue
    for candidate in _detect_clusters(beam):
        enqueue(candidate)

    # Get the detection candidates
    return queue.get_candidates(beam.id)

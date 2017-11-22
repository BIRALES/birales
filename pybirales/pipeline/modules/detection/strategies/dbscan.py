import itertools
import logging as log
import os
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
from astropy.time import Time, TimeDelta
from pybirales import settings
from pybirales.pipeline.modules.detection.beam import Beam
from pybirales.pipeline.modules.detection.detection_clusters import DetectionCluster
from pybirales.pipeline.modules.detection.strategies.strategies import SpaceDebrisDetectionStrategy
from pybirales.plotters.spectrogram_plotter import plotter
from sklearn import linear_model
from sklearn.cluster import DBSCAN


def dd2(beam):
    # Apply the pre-processing filters to the beam data
    candidates = []
    try:
        # beam.apply_filters()
        # Run detection algorithm on the beam data to extract possible candidates
        candidates = DBScanDetectionStrategy().detect_space_debris_candidates(beam)
    except Exception:
        log.exception('Something went wrong with process')
    return candidates


class DBScanDetectionStrategy(SpaceDebrisDetectionStrategy):
    name = 'Spirit'
    _eps = 5
    _min_samples = 10
    _algorithm = 'kd_tree'
    _r2_threshold = 0.90
    _merge_threshold = 0.30

    def __init__(self):
        SpaceDebrisDetectionStrategy.__init__(self)
        # Initialise the clustering algorithm
        self.db_scan = DBSCAN(eps=self._eps,
                              min_samples=self._min_samples,
                              algorithm=self._algorithm,
                              # metric="precomputed",
                              n_jobs=-1)
        # self.db_scan = hdbscan.HDBSCAN(min_cluster_size=self._min_samples)

        self._thread_pool = ThreadPool(settings.detection.nthreads)

        # self._m_pool = Pool()

        self._linear_model = linear_model.RANSACRegressor(linear_model.LinearRegression())

    def detect(self, obs_info, input_data):
        beams = [Beam(beam_id=n_beam, obs_info=obs_info, beam_data=input_data)
                 for n_beam in range(settings.detection.beam_range[0], settings.detection.beam_range[1])]

        # Process the beam data to detect the beam candidates
        if settings.detection.multi_proc:
            new_beam_candidates = self._get_beam_candidates_multi_process(beams)
        elif settings.detection.nthreads > 1:
            new_beam_candidates = self._get_beam_candidates_parallel(beams)
        else:
            new_beam_candidates = self._get_beam_candidates_single(beams)

        return new_beam_candidates

    def c(self, data):
        cluster_labels = self.db_scan.fit_predict(data)
        return cluster_labels

    def c2(self, data):
        e, l = DB2.DBSCAN(data, minPts=5, eps=3.0, verbose=True)
        return l

    def _create_clusters(self, beam):
        """
        Use the DBScan algorithm to create a set of clusters from the given beam data
        :param beam: The beam object from which the clusters will be generated
        :return:
        """
        # Select the data points that are non-zero and transform them in a time (x), channel (y) nd-array
        data = np.column_stack(np.where(beam.snr > 0))

        if not np.any(data):
            return []

        # X = euclidean_distances(data, data)
        X = data
        # Perform clustering on the data and returns cluster labels the points are associated with
        try:

            cluster_labels = self.c(X)

        # cluster_labels = self.c2(X)
        except ValueError:
            log.exception('DBSCAN failed in beam %s', beam.id)
            return []
        # plotter.plot(beam.snr, 'detection/db_scan/' + str(time.time()), beam.id == 0,
        #              cluster_labels=cluster_labels)

        # Select only those labels which were not classified as noise (-1)
        filtered_cluster_labels = cluster_labels[cluster_labels > -1]

        log.debug('Process %s: beam %s, %s out of %s data points was considered to be noise',
                  os.getpid(),
                  beam.id,
                  len(cluster_labels) - len(filtered_cluster_labels),
                  len(cluster_labels))

        ref_time = Time(beam.t_0)
        dt = TimeDelta(beam.dt, format='sec')

        log.debug('%s unique clusters were detected', len(np.unique(filtered_cluster_labels)))

        # Group the data points in clusters
        clusters = []
        for label in np.unique(filtered_cluster_labels):
            data_indices = data[np.where(cluster_labels == label)]

            log.debug('beam %s: cluster %s contains %s data points', beam.id, label, len(data_indices[:, 1]))

            channel_indices = data_indices[:, 1]
            time_indices = data_indices[:, 0]

            # If cluster is 'small' do not consider it
            if len(channel_indices) < self._min_samples:
                log.debug('Ignoring small cluster with %s data points', len(channel_indices))
                continue

            # Create a Detection Cluster from the cluster data
            cluster = DetectionCluster(model=self._linear_model,
                                       beam=beam,
                                       time_data=[ref_time + t * dt for t in beam.time[time_indices]],
                                       channels=beam.channels[channel_indices],
                                       snr=beam.snr[(time_indices, channel_indices)])

            # Add only those clusters that are linear
            if cluster.is_linear(threshold=0.9):
                log.debug('Cluster with m:%3.2f, c:%3.2f, n:%s and r:%0.2f is considered to be linear.', cluster.m,
                          cluster.c, len(channel_indices), cluster._score)
                clusters.append(cluster)

        log.debug('DBSCAN detected %s clusters in beam %s, of which %s are linear',
                  len(np.unique(filtered_cluster_labels)),
                  beam.id, len(clusters))

        return clusters

    def _merge_clusters(self, clusters):
        """
        Merge clusters based on how similar the gradient and y-intercept are

        :param clusters: The clusters that will be evaluated for merge
        :type clusters: list of DetectionCandidates
        :return:
        """

        if not clusters:
            log.debug("No clusters to merge")
            return clusters

        done = False
        while not done:
            done = True
            for cluster_1, cluster_2 in itertools.combinations(clusters, 2):
                if cluster_1.is_similar_to(cluster_2, self._merge_threshold):
                    # Create a new merged cluster if clusters are similar
                    merged_cluster = cluster_1.merge(cluster_2)

                    # Append the new cluster to the list
                    clusters.append(merged_cluster)

                    # Delete old clusters
                    clusters = [cluster for cluster in clusters if cluster not in [cluster_1, cluster_2]]

                    # Re-compare the cluster with the rest
                    done = False

                    # Break loop to re-compare
                    break

        return clusters

    def _get_beam_candidates_single(self, beams):
        """
        Run the detection algorithm using 1 process

        :return: beam_candidates Beam candidates detected across the 32 beams
        """

        log.debug('Running space debris detection algorithm on %s beams in serial', len(beams))

        # Get the detected beam candidates
        beam_candidates = [self.detect_space_debris_candidates(beam) for beam in beams]

        # Do not add beam candidates that a
        return [candidate for sub_list in beam_candidates for candidate in sub_list if candidate]

    def _get_beam_candidates_multi_process(self, beams):
        beam_candidates = []

        # pool = self._m_pool
        _m_pool = Pool()
        try:

            beam_candidates = _m_pool.map(dd2, beams)
            # results = [pool.apply_async(dd2, args=(beam,)) for beam in beams]
            # beam_candidates = [p.get() for p in results]
        except Exception:
            log.exception('An exception has occurred')

        # self.pool.close()
        # self.pool.join()

        # Flatten list of beam candidates returned by the N threads
        return [candidate for sub_list in beam_candidates for candidate in sub_list if candidate]

    def _get_beam_candidates_parallel(self, beams):
        """
        Run the detection algorithm using N threads

        :return: beam_candidates Beam candidates detected across the 32 beams
        """

        log.debug('Running space debris detection algorithm on %s beams in parallel', len(beams))

        # Run using N threads
        beam_candidates = self._thread_pool.map(self.detect_space_debris_candidates, beams)
        # Flatten list of beam candidates returned by the N threads
        return [candidate for sub_list in beam_candidates for candidate in sub_list if candidate]

    def detect_space_debris_candidates(self, beam):

        plotter.plot(beam, 'detection/input_beam/' + str(beam.id) + '_' + str(beam.t_0), beam.id == 0)

        # Apply the pre-processing filters to the beam data
        beam.apply_filters()

        plotter.plot(beam, 'detection/filtered_beam/' + str(beam.id) + '_' + str(beam.t_0), beam.id == 0)

        # Run detection algorithm on the beam data to extract possible candidates
        clusters = self._create_clusters(beam)

        # Merge clusters which are identified as being similar
        merged_clusters = self._merge_clusters(clusters)

        return merged_clusters

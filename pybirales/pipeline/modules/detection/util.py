import logging as log

import pandas as pd
from scipy import stats

from msds.util import timeit
from msds.visualisation import *
from pybirales import settings
from pybirales.events.events import TrackTransittedEvent
from pybirales.events.publisher import publish
from pybirales.pipeline.modules.detection.exceptions import DetectionClusterIsNotValid
from pybirales.pipeline.modules.detection.space_debris_candidate import SpaceDebrisTrack
from pybirales.services.post_processing.tdm_persister import persist


def apply_doppler_mask(doppler_mask, channels, doppler_range, obs_info):
    """

    :param obs_info:
    :return:
    """

    if doppler_mask is None:
        channels = np.arange(obs_info['start_center_frequency'],
                             obs_info['start_center_frequency'] + obs_info['channel_bandwidth'] * obs_info[
                                 'nchans'], obs_info['channel_bandwidth'])

        a = obs_info['transmitter_frequency'] + doppler_range[0] * 1e-6
        b = obs_info['transmitter_frequency'] + doppler_range[1] * 1e-6

        doppler_mask = np.bitwise_and(channels < b, channels > a)

        # print a
        # print b
        # print len(channels)
        # print np.argmin(channels < b)
        # print np.argmax(channels > a)
        # print channels[2392], channels[6502]

        channels = channels[doppler_mask]

    return channels, doppler_mask


def _debug_msg(cluster, iter_count):
    """

    :param cluster:
    :return:
    """
    m, c, r_value, _, _ = stats.linregress(cluster['channel_sample'], cluster['time_sample'])
    return '{:03} (m={:0.2f}, c={:0.2f}, s={:0.2f}, n={}, i={})'.format(id(cluster) % 100, m, c, r_value,
                                                                        cluster.shape[0],
                                                                        iter_count)


# @timeit
def aggregate_clusters_old(candidates, new_clusters, obs_info, notifications=False, save_candidates=False):
    """
    Create Space Debris Tracks from the detection clusters identified in each beam

    :param clusters:
    :return:
    """

    for cluster in new_clusters:
        for candidate in candidates:
            # If beam candidate is similar to candidate, merge it.
            if candidate.is_parent_of(cluster):
                try:
                    candidate.add(cluster)
                except DetectionClusterIsNotValid:
                    log.debug('Beam candidate {} could not be added to track {:03}'.format(
                        _debug_msg(cluster, obs_info['iter_count']), id(candidate) % 1000))
                else:
                    log.debug(
                        'Beam candidate {} added to track {:03}'.format(_debug_msg(cluster, obs_info['iter_count']),
                                                                        id(candidate) % 1000))

                    break
        else:
            # Beam cluster does not match any candidate. Create a new candidate track from it.
            try:
                sd = SpaceDebrisTrack(obs_info=obs_info, cluster=cluster)
            except DetectionClusterIsNotValid:
                continue

            log.debug('Created new track {} from Beam candidate {}'.format(id(sd),
                                                                           _debug_msg(cluster, obs_info['iter_count'])))

            # Add the space debris track to the candidates list
            candidates.append(sd)

    # Notify the listeners, that a new detection was made
    if notifications:
        [candidate.send_notification() for candidate in candidates]

    # Save candidates that were updated
    if save_candidates:
        [candidate.save() for candidate in candidates if not candidate.saved]

    return candidates


@timeit
def data_association(tracks, tentative_tracks, obs_info, notifications=False, save_candidates=False):
    """
    Create Space Debris Tracks from the detection clusters identified in each beam

    :param clusters:
    :return:
    """

    for tentative_track in tentative_tracks:
        for track in tracks:
            # do not compare with cancelled or terminated tracks
            if track.cancelled or track.terminated:
                continue

            if track.is_parent_of(tentative_track):
                try:
                    track.associate(tentative_track)
                except DetectionClusterIsNotValid:
                    # Upon association, the track is no longer valid
                    pass
                else:
                    # no need to check the other candidates. Tentative track is added to the first matching track
                    break
        else:
            # Beam cluster does not match any candidate. Create a new candidate track from it.
            try:
                new_track = SpaceDebrisTrack(obs_info=obs_info, cluster=tentative_track)
            except DetectionClusterIsNotValid:
                continue

            log.debug('Created new track {} from Beam candidate {}'.format(id(new_track),
                                                                           _debug_msg(tentative_track,
                                                                                      obs_info['iter_count'])))

            # Add the space debris track to the candidates list
            tracks.append(new_track)

    # Notify the listeners, that a new detection was made
    if notifications:
        [track.send_notification() for track in tracks]

    # Save candidates that were updated
    if save_candidates:
        [track.save() for track in tracks if not track.saved]

    return tracks


@timeit
def active_tracks(obs_info, tracks, iter_count):
    """

    :param candidates:
    :param iter_count:
    :return:
    """

    for track in tracks:
        valid, reason = track.is_valid()
        if not valid:
            track.cancel()
            # delete from database but keep it in memory
            log.info("Track {} was cancelled in iteration {}. Reason: {}".format(track.id, iter_count, reason))
            continue

        if track.track_expired(obs_info):
            log.info("Track {} was terminated in iteration {}".format(track.id, iter_count))
            track.terminate()

            publish(TrackTransittedEvent(track))

            if settings.detection.save_tdm:
                persist(obs_info, track, debug=settings.detection.debug_candidates)

    return tracks


def create_candidate(cluster_data, channels, iter_count, n_samples, t0, td, channel_noise, beam_id):
    channel_ndx = cluster_data[:, 0].astype(int)
    time_ndx = cluster_data[:, 1].astype(int)
    time_sample = time_ndx + iter_count * n_samples
    channel = channels[channel_ndx]

    beam_ids = np.full(time_ndx.shape[0], beam_id)
    noise_est = np.mean(channel_noise[beam_id])
    snr = 10 * np.log10(cluster_data[:, 2] / noise_est)

    return pd.DataFrame({
        'time_sample': time_sample,
        'channel_sample': channel_ndx,
        'time': t0 + (time_sample - n_samples * iter_count) * td,
        'channel': channel,
        'snr': snr,
        'beam_id': beam_ids,
        'iter': np.full(time_ndx.shape[0], iter_count),
    })

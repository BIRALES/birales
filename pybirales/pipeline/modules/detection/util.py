import logging as log

import pandas as pd
from scipy import stats

from msds.visualisation import *
from pybirales.events.events import TrackTransittedEvent
from pybirales.events.publisher import publish
from pybirales.pipeline.modules.detection.exceptions import DetectionClusterIsNotValid
from pybirales.pipeline.modules.detection.space_debris_candidate import SpaceDebrisTrack
from pybirales.services.post_processing.tdm_persister import persist


def __id(obj):
    return id(obj) % 1000


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
        # print doppler_range
        # print channels[2392], channels[6502]

        # print "tx is at:", np.argmin(channels <= obs_info['transmitter_frequency'])
        # print "tx is at:", np.argmax(channels <= obs_info['transmitter_frequency'])
        # print "tx is at:", np.argmin(channels >= obs_info['transmitter_frequency'])
        # print "tx is at:", np.argmax(channels >= obs_info['transmitter_frequency'])

        channels = channels[doppler_mask]

        # print min(channels), max(channels), len(channels)



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
def data_association(tracks, tentative_tracks, obs_info, notifications=False, save_candidates=False):
    """
    Create Space Debris Tracks from the detection clusters identified in each beam

    :param clusters:
    :return:
    """
    # print len(tentative_tracks), len(tracks)
    for i, tentative_track in enumerate(tentative_tracks):
        for j, track in enumerate(tracks):
            # print "Comparing track {} with tentative {}".format(__id(track), __id(tentative_track))
            # do not compare with cancelled or terminated tracks
            if track.cancelled or track.terminated:
                # print "track {} was terminated or cancelled {} {}".format(__id(track), track.cancelled, track.cancelled)
                continue

            if track.is_parent_of(tentative_track):

                associated = track.associate(tentative_track)

                # print "track {} associated with cluster {}".format(__id(track), __id(tentative_track))

                if associated:
                    break
            # else:
            #     print "Track {} and tentative {} do not match".format(__id(track), __id(tentative_track))

        else:
            # Beam cluster does not match any candidate. Create a new candidate track from it.
            try:
                new_track = SpaceDebrisTrack(obs_info=obs_info, cluster=tentative_track)
            except DetectionClusterIsNotValid:
                continue

            # print "Created new track {} m={} from {}".format(__id(new_track), new_track.m, __id(tentative_track))
            log.info('New track initiated {}'.format(__id(new_track)))

            # Add the space debris track to the candidates list
            tracks.append(new_track)

    # Notify the listeners, that a new detection was made
    if notifications:
        [track.send_notification() for track in tracks]

    # Save candidates that were updated
    if save_candidates:
        [track.save() for track in tracks if not track.saved]

    return tracks


# @timeit
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
            log.info("Track {} was cancelled in iteration {}. Reason: {}".format(track.id % 1000, iter_count, reason))
            continue

        if track.track_expired(obs_info):
            log.info("Track {} was terminated in iteration {}".format(track.id % 1000, iter_count))
            track.terminate()

            publish(TrackTransittedEvent(track))

            persist(obs_info, track)

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

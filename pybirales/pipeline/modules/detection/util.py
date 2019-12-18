import logging as log

from scipy import stats

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


def aggregate_clusters(candidates, clusters, obs_info, notifications=False, save_candidates=False):
    """
    Create Space Debris Tracks from the detection clusters identified in each beam

    :param clusters:
    :return:
    """

    for cluster in clusters:
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


def active_tracks(obs_info, candidates, n_rso, iter_count):
    """

    :param candidates:
    :param iter_count:
    :return:
    """

    temp_candidates = []
    # Tracks that were deleted should not count as 'track transitted'
    invalid_tracks = 0
    transitted = 0
    for candidate in candidates:
        if not candidate.is_valid():
            candidate.delete()

            invalid_tracks += 1
            continue

        if candidate.track_expired(obs_info):
            # print iter_count

            # if candidate.has_transitted(iter_count=iter_count):
            # If the candidate is not valid delete it else it won't be added to the list
            # if not candidate.is_valid:
            #     candidate.delete()
            # else:

            # Track has transitted outside the field of view of the instrument
            publish(TrackTransittedEvent(candidate))
            transitted += 1

            persist(obs_info, candidate, debug=settings.detection.debug_candidates)
        else:
            temp_candidates.append(candidate)

    # transitted = len(self._candidates) - len(temp_candidates) - invalid_tracks
    n_rso += transitted
    log.info('Result: {} tracks have transitted. {} tracks are currently in detection window.'.format(transitted,
                                                                                                      len(
                                                                                                          temp_candidates)))

    return temp_candidates, n_rso

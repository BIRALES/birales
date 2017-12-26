#!/usr/bin/python
import datetime
import numpy as np
import ephem
import pytz

# Define instrument position and list calibration sources
latitude = 44.523733
longitude = 11.645929
calibration_sources = {'casa': {'name': '3C461', 'ra': '23:21:10.6', 'dec': '+58:33.1', 'flux': 11000.0},
                       'virA': {'name': '3C274', 'ra': '12:28:18.0', 'dec': '+12:40.1', 'flux': 970.0},
                       'tauA': {'name': '3C144', 'ra': '05:31:30.0', 'dec': '+21:58.4', 'flux': 1420.0},
                       'cygnus': {'name': '3C405', 'ra': '19:57:45.3', 'dec': '+40:36.0', 'flux': 8100.0}}

# Define observer
obs = ephem.Observer()
obs.long = longitude * (ephem.pi / 180.)
obs.lat = latitude * (ephem.pi / 180.)
obs.epoch = 2000.0
epoch = 1950


def get_calibration_sources():
    """
    Return a list of known calibration sources

    :return:
    """

    return calibration_sources.keys()


def get_calibration_source_declination(source_name):
    """
    Get the declination of the calibration source

    :param source_name:
    :return:
    """

    # Check whether calibration source is valid
    if source_name not in calibration_sources.keys():
        raise Exception("{} not a known calibration source".format(source_name))

    c_v = calibration_sources[source_name]

    return np.math.degrees(ephem.degrees(c_v['dec']))


def _ephem_compute(source_name, obs_date=None):
    """
    Compute the ephem body

    :param source_name:
    :param obs_date:
    :return:
    """

    # Check whether calibration source is valid
    if source_name not in calibration_sources.keys():
        raise Exception("{} not a known calibration source".format(source_name))

    if obs_date is None:
        obs_date = ephem.julian_date()
    else:
        obs_date = ephem.julian_date(obs_date)
    obs.date = ephem.date(obs_date - 2415020)

    # Calculate next transit time
    c_v = calibration_sources[source_name]

    ephem_line = '%s,f,%s,%s,%s,%d' % (c_v['name'], c_v['ra'], c_v['dec'], np.log10(c_v['flux']), epoch)
    body = ephem.readdb(ephem_line)
    body.compute(obs)

    return body


def get_next_transit(source_name, obs_date=None):
    """
    Get the first transit time of a source after an obs_date

    :param source_name:
    :param obs_date:
    :return:
    """
    body = _ephem_compute(source_name, obs_date)
    next_transit = obs.next_transit(body)
    time_to_transit = next_transit - obs.date

    return next_transit, time_to_transit * 24 * 3600


def get_previous_transit(source_name, obs_date=None):
    """
    Get the last transit time of a source before an obs_date

    :param source_name:
    :param obs_date:
    :return:
    """
    body = _ephem_compute(source_name, obs_date)
    previous_transit = obs.previous_transit(body)
    time_to_transit = previous_transit - obs.date

    return previous_transit.datetime().replace(tzinfo=pytz.utc), time_to_transit * 24 * 3600


def get_best_calibration_obs(from_date, to_date, time_to_calibrate):
    """

    Return the possible/available calibration sources for a future observation date

    :param from_date:
    :param to_date:
    :param time_to_calibrate:
    :return: A dictionary of available sources together with their parameters
    """
    if from_date is None:
        from_date = to_date - datetime.timedelta(days=2)

    if to_date <= from_date:
        raise ValueError("TO date cannot be before FROM date")

    # Account for time to calibrate
    max_to_date = to_date - time_to_calibrate

    available_calibration_sources = []
    for source, source_parameters in calibration_sources.iteritems():
        previous_transit_time, ttt = get_previous_transit(source, max_to_date)

        if from_date < previous_transit_time < max_to_date:
            source_parameters['name'] = source
            source_parameters['transit_time'] = previous_transit_time

            available_calibration_sources.append(source_parameters)

    return available_calibration_sources


if __name__ == "__main__":
    for k, v in calibration_sources.iteritems():
        date, seconds = get_next_transit(k, datetime.datetime.now())
        print "%+8s : RA: %11s  DEC: %11s  FLUX: %8s  TRANSIT (UTC): %-20s" % (
            k,
            v['ra'],
            v['dec'],
            v['flux'],
            date)

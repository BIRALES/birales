#!/usr/bin/python
import datetime
import numpy as np
import ephem

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
    """ Return a list of known calibration sources"""
    return calibration_sources.keys()


def get_calibration_source_declination(source_name):
    """Get calibration source declination"""

    # Check whether calibration source is valid
    if source_name not in calibration_sources.keys():
        raise Exception("{} not a known calibration source".format(source_name))

    v = calibration_sources[source_name]
    return np.math.degrees(ephem.degrees(v['dec']))


def _ephem_compute(source_name, date=None):
    """ Get the next transit time for the give source"""

    # Check whether calibration source is valid
    if source_name not in calibration_sources.keys():
        raise Exception("{} not a known calibration source".format(source_name))

    if date is None:
        date = ephem.julian_date()
    else:
        date = ephem.julian_date(date)
    obs.date = ephem.date(date - 2415020)

    # Calculate next transit time
    v = calibration_sources[source_name]

    ephemline = '%s,f,%s,%s,%s,%d' % (v['name'], v['ra'], v['dec'], np.log10(v['flux']), epoch)
    body = ephem.readdb(ephemline)
    body.compute(obs)

    return body


def get_next_transit(source_name, date=None):
    body = _ephem_compute(source_name, date)
    next_transit = obs.next_transit(body)
    time_to_transit = next_transit - obs.date

    return next_transit, time_to_transit * 24 * 3600


def get_previous_transit(source_name, date=None):
    body = _ephem_compute(source_name, date)
    previous_transit = obs.previous_transit(body)
    time_to_transit = previous_transit - obs.date

    return previous_transit.datetime(), time_to_transit * 24 * 3600


def get_best_calibration_obs(observation_date):
    min_time_to_transit = datetime.timedelta(minutes=30)
    max_flux = -1
    chosen_obs = None
    for source, v in calibration_sources.iteritems():
        previous_transit_time, ttt = get_previous_transit(source, observation_date)

        if previous_transit_time > observation_date:
            continue

        if v['flux'] > max_flux and min_time_to_transit < (observation_date - previous_transit_time):
            max_flux = v['flux']
            chosen_obs = {
                'name': source,
                'parameters': v,
                'transit_time': previous_transit_time
            }

    return chosen_obs


if __name__ == "__main__":
    for k, v in calibration_sources.iteritems():
        date, seconds = get_next_transit(k, datetime.datetime.now())
        print "%+8s : RA: %11s  DEC: %11s  FLUX: %8s  TRANSIT (UTC): %-20s" % (
            k,
            v['ra'],
            v['dec'],
            v['flux'],
            date)

import datetime
import json
import time

import ephem
import spacetrack.operators as op
from spacetrack import SpaceTrackClient

USERNAME = 'denis0@gmail.com'
PASSWORD = 'spac3_track_PASS'
BIRALES_FOV_RA = 2
BIRALES_FOV_DEC = 2
BIRALES_LON = 11.6459889
BIRALES_LAT = 44.52357778

obs = ephem.Observer()
obs.long = BIRALES_LON
obs.lat = BIRALES_LAT


def get_ra_inc_range(timestamp, az, el):
    obs.date = timestamp
    return obs.radec_of(str(az), str(el))


def rate_limiter(until):
    duration = int(round(until - time.time()))
    print('Sleeping for {:d} seconds.'.format(duration))


# obs.pressure = 0 #No refraction
now = datetime.datetime.now()

ra, dec = get_ra_inc_range(timestamp=now, az=0, el=90)

from skyfield import api
from astropy.time import Time

BIRALES_LON = 11.6459889
BIRALES_LAT = 44.52357778
ts = api.load.timescale()

atime = Time('2019-11-22T16:06:06.7940', scale='utc')
atime = Time.now()

t = ts.from_astropy(atime)
topos = api.Topos(latitude_degrees=BIRALES_LAT, longitude_degrees=BIRALES_LON)
topos_pos = topos.at(t)
pos = topos_pos.from_altaz(alt_degrees=90., az_degrees=0)

ra, dec, distance = pos.radec()
print(dec.degrees, ra._degrees)

ra_min = ra._degrees - BIRALES_FOV_RA
ra_max = ra._degrees + BIRALES_FOV_RA
dec_min = dec.degrees - BIRALES_FOV_DEC
dec_max = dec.degrees + BIRALES_FOV_DEC

print "Birales Telescope is currently pointed towards: RA: {} deg, DEC: {} deg".format(ra, dec)
print "FoV range is set to: RA: {}-{} deg, DEC: {}-{} deg".format(ra_min, ra_max, dec_min, dec_max)

st = SpaceTrackClient(identity=USERNAME, password=PASSWORD)
st.callback = rate_limiter

data = st.tle_latest(ordinal=1, epoch='>now-30',
                     inclination=op.inclusive_range(dec_min, dec_max),
                     ra_of_asc_node=op.inclusive_range(ra_min, ra_max), format='json')

data_jsn = json.loads(data)

for i, tle in enumerate(data_jsn):
    print i, "NORAD ID {}: RA: {} deg, INC: {} deg, EPOCH: {}".format(
        tle['NORAD_CAT_ID'],
        tle['RA_OF_ASC_NODE'],
        tle['INCLINATION'],
        tle['EPOCH'],
    )

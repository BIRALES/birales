import datetime
import logging
import warnings

import numpy as np
from astropy import constants
from astropy import units as u
from astropy.coordinates import Angle, EarthLocation, SkyCoord, AltAz
from astropy.time import Time
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyWarning

# Mute Astropy Warnings
warnings.simplefilter('ignore', category=AstropyWarning)


config = {}

config['reference_antenna_location'] = [44.52357778, 11.6459889]
config['antenna_locations'] = [[-8.499750, -35.000000, 0.000000],
                               [-2.833250, -35.000000, 0.000000],
                               [2.833250, -35.000000, 0.000000],
                               [8.499750, -35.000000, 0.000000],
                               [-8.499750, -25.000000, 0.000000],
                               [-2.833250, -25.000000, 0.000000],
                               [2.833250, -25.000000, 0.000000],
                               [8.499750, -25.000000, 0.000000],
                               [-8.499750, -15.000000, 0.000000],
                               [-2.833250, -15.000000, 0.000000],
                               [2.833250, -15.000000, 0.000000],
                               [8.499750, -15.000000, 0.000000],
                               [-8.499750, -5.000000, 0.000000],
                               [-2.833250, -5.000000, 0.000000],
                               [2.833250, -5.000000, 0.000000],
                               [8.499750, -5.000000, 0.000000],
                               [-8.499750, 5.000000, 0.000000],
                               [-2.833250, 5.000000, 0.000000],
                               [2.833250, 5.000000, 0.000000],
                               [8.499750, 5.000000, 0.000000],
                               [-8.499750, 15.000000, 0.000000],
                               [-2.833250, 15.000000, 0.000000],
                               [2.833250, 15.000000, 0.000000],
                               [8.499750, 15.000000, 0.000000],
                               [-8.499750, 25.000000, 0.000000],
                               [-2.833250, 25.000000, 0.000000],
                               [2.833250, 25.000000, 0.000000],
                               [8.499750, 25.000000, 0.000000],
                               [-8.499750, 35.000000, 0.000000],
                               [-2.833250, 35.000000, 0.000000],
                               [2.833250, 35.000000, 0.000000],
                               [8.499750, 35.000000, 0.000000]]

config['pointings'] = [[0, 0]]
config['reference_pointing'] = [0, 64.515]
config['nbeams'] = len(config['pointings'])
config['start_center_frequency'] = 410.03125
config['channel_bandwidth'] = 0.078125


class Pointing(object):
    """ Pointing class which periodically updates pointing weights """

    def __init__(self, config, nsubs, nants):

        # Call superclass initialiser
        super(Pointing, self).__init__()

        # Make sure that we have enough antenna locations
        if len(config['antenna_locations']) != nants:
            print("Pointing: Mismatch between number of antennas and number of antenna locations")

        # Make sure that we have enough pointing
        if len(config['pointings']) != config['nbeams']:
           print("Pointing: Mismatch between number of beams and number of beam pointings")

        # Initialise Pointing
        array = config['antenna_locations']
        self._start_center_frequency = config['start_center_frequency']
        self._reference_location = config['reference_antenna_location']
        self._reference_pointing = config['reference_pointing']
        self._pointings = config['pointings']
        self._bandwidth = config['channel_bandwidth']
        self._nbeams = config['nbeams']
        self._nants = nants
        self._nsubs = nsubs

        # Create AntennaArray object
        self._array = AntennaArray(array, self._reference_location[0], self._reference_location[1])

        # Calculate displacement vectors to each antenna from reference antenna
        # (longitude, latitude)
        self._reference_location = EarthLocation.from_geodetic(self._reference_location[1],
                                                               self._reference_location[0],
                                                               height=self._array.height)

        self._vectors_enu = np.full([self._nants, 3], np.nan)
        for i in range(self._nants):
            self._vectors_enu[i, :] = self._array.antenna_position(i)

        # Create initial weights
        self.weights = np.ones((self._nsubs, self._nbeams, self._nants), dtype=np.complex64)

        # Ignore AstropyWarning
        warnings.simplefilter('ignore', category=AstropyWarning)

    def run(self, pointing_time=None):
        """ Run pointing """

        # Get time once (so that all beams use the same time for pointing)
        if pointing_time is None:
            pointing_time = Time(datetime.datetime.utcnow(), scale='utc', location=self._reference_location)
        else:
            pointing_time = Time(pointing_time, scale='utc', location=self._reference_location)

        for beam in range(self._nbeams):
            self.point_array(beam, self._reference_pointing[0], self._reference_pointing[1],
                             self._pointings[beam][0], self._pointings[beam][1],
                             pointing_time)

        logging.info("Updated beamforming coefficients")

    def point_array_static(self, beam, altitude, azimuth):
        """ Calculate the phase shift given the altitude and azimuth coordinates of a sky object as astropy angles
        :param beam: beam to which this applies
        :param altitude: altitude coordinates of a sky object as astropy angle
        :param azimuth: azimuth coordinates of a sky object as astropy angles
        :return: The phase shift in radians for each antenna
        """

        for i in range(self._nsubs):
            # Calculate complex coefficients
            frequency = Quantity(self._start_center_frequency + (i * self._bandwidth / self._nsubs), u.MHz)
            real, imag = self._phaseshifts_from_altitude_azimuth(altitude.rad, azimuth.rad, frequency,
                                                                 self._vectors_enu)

            # Apply to weights file
            self.weights[i, beam, :].real = real
            self.weights[i, beam, :].imag = imag

    def point_array(self, beam, ref_ra, ref_dec, delta_ra, delta_dec, pointing_time=None):
        """ Calculate the phase shift between two antennas which is given by the phase constant (2 * pi / wavelength)
        multiplied by the projection of the baseline vector onto the plane wave arrival vector
        :param beam: Beam to which this applies
        :param right_ascension: Right ascension of source (astropy angle, or string that can be converted to angle)
        :param declination: Declination of source (astropy angle, or string that can be converted to angle)
        :param pointing_time: Time of observation (in format astropy time)
        :return: The phaseshift in radians for each antenna
        """

        # Type conversions if required
        ref_ra = Angle(ref_ra, u.deg)
        ref_dec = Angle(ref_dec, u.deg)
        delta_ra = Angle(delta_ra, u.deg)
        delta_dec = Angle(delta_dec, u.deg)

        # If we're using static beam, then we must adjust RA (by setting reference RA equal to LST)
       # ref_ra = Angle(pointing_time.sidereal_time('apparent'))

        # Convert RA DEC to ALT AZ
        alt, az = self._ra_dec_to_alt_az(ref_ra + delta_ra,
                                         ref_dec + delta_dec,
                                         Time(pointing_time),
                                         self._reference_location)

        # Point beam to required ALT AZ
       # print("RA: {}, DEC: {}, ALT: {}, AZ: {}".format(ref_ra + delta_ra, ref_dec + delta_dec, alt, az))
        self.point_array_static(beam, alt, az)

    @staticmethod
    def _ra_dec_to_alt_az(right_ascension, declination, time, location):
        """ Calculate the altitude and azimuth coordinates of a sky object from right ascension and declination and time
        :param right_ascension: Right ascension of source (in astropy Angle on string which can be converted to Angle)
        :param declination: Declination of source (in astropy Angle on string which can be converted to Angle)
        :param time: Time of observation (as astropy Time")
        :param location: astropy EarthLocation
        :return: Array containing altitude and azimuth of source as astropy angle
        """

        # Initialise SkyCoord object using the default frame (ICRS) and convert to horizontal
        # coordinates (altitude/azimuth) from the antenna's perspective.
        sky_coordinates = SkyCoord(ra=right_ascension, dec=declination, location=location)
        c_altaz = sky_coordinates.transform_to(AltAz(obstime=time, location=location))

        return c_altaz.alt, c_altaz.az

    @staticmethod
    def _phaseshifts_from_altitude_azimuth(altitude, azimuth, frequency, displacements):
        """
        Calculate the phaseshift using a target altitude Azimuth
        :param altitude: The altitude of the target astropy angle
        :param azimuth: The azimuth of the target astropy angle
        :param frequency: The frequency of the observation as astropy quantity.
        :param displacements: Numpy array: The displacement vectors between the antennae in East, North, Up
        :return: The phaseshift angles in radians
        """
        scale = np.array(
            [-np.sin(azimuth) * np.cos(altitude), -np.cos(azimuth) * np.cos(altitude), -np.sin(altitude)])

        path_length = np.dot(scale, displacements.transpose())

        k = (2.0 * np.pi * frequency.to(u.Hz).value) / constants.c.value
        return np.cos(np.multiply(k, path_length)), np.sin(np.multiply(k, path_length))


# --------------------------------------------------------------------------------------------------

class AntennaArray(object):
    """ Class representing antenna array """

    def __init__(self, positions, ref_lat, ref_long):
        self._ref_lat = ref_lat
        self._ref_lon = ref_long
        self._ref_height = 0

        self._x = None
        self._y = None
        self._z = None
        self._height = None

        self._n_antennas = None

        # If positions is defined, initialise array
        self.load_from_positions(positions)

    def positions(self):
        """
        :return: Antenna positions
        """
        return self._x, self._y, self._z, self._height

    def antenna_position(self, i):
        """
        :return: Position for antenna i
        """
        return self._x[i], self._y[i], self._z[i]

    @property
    def size(self):
        return self._n_antennas

    @property
    def lat(self):
        """ Return reference latitude
        :return: reference latitude
        """
        return self._ref_lat

    @property
    def lon(self):
        """ Return reference longitude
        :return: reference longitude
        """
        return self._ref_lon

    @property
    def height(self):
        """
        Return reference height
        :return: reference height
        """
        return self._ref_height

    def load_from_positions(self, positions):
        """ Create antenna array from positions
        :param positions: Array of antenna positions
        """

        # Keep track of how many antennas are in the array
        self._n_antennas = len(positions)

        # Convert file data to astropy format, and store lat, lon and height for each antenna
        self._x = [positions[i][0] for i in range(len(positions))]
        self._y = [positions[i][1] for i in range(len(positions))]
        self._z = [positions[i][2] for i in range(len(positions))]
        self._height = [0 for i in range(len(positions))]


if __name__ == "__main__":

    # Should be pointing to zenith (regardless of time)
    for i in range(-90, 90, 2):
        config['reference_antenna_location'] = [44.52357778, 11.6459889]
        config['reference_pointing'] = [0,  i]
        config['pointings'] = [[0, 0]]

        pointing = Pointing(config, 1, 32)
        pointing.run()
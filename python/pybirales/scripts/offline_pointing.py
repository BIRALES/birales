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

config['reference_antenna_location'] = [11.6459889,  44.523880962]
config['antenna_locations'] = [[0,      0,  0],
                               [5.6665, 0,  0],
                               [11.333, 0,  0],
                               [16.999, 0,  0],
                               [0,      10, 0, ],
                               [5.6665, 10, 0],
                               [11.333, 10, 0],
                               [16.999, 10, 0],
                               [0,      20, 0],
                               [5.6665, 20, 0],
                               [11.333, 20, 0],
                               [16.999, 20, 0],
                               [0,      30, 0],
                               [5.6665, 30, 0],
                               [11.333, 30, 0],
                               [16.999, 30, 0],
                               [0,      40, 0],
                               [5.6665, 40, 0],
                               [11.333, 40, 0],
                               [16.999, 40, 0],
                               [0,      50, 0],
                               [5.6665, 50, 0],
                               [11.333, 50, 0],
                               [16.999, 50, 0],
                               [0,      60, 0],
                               [5.6665, 60, 0],
                               [11.333, 60, 0],
                               [16.999, 60, 0],
                               [0,      70, 0],
                               [5.6665, 70, 0],
                               [11.333, 70, 0],
                               [16.999, 70, 0]]

config['pointings'] = [[0, 0]]
config['reference_declination'] = 64.515
config['nbeams'] = len(config['pointings'])
config['start_center_frequency'] = 410
config['channel_bandwidth'] = 0.0001 # 0.078125


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
        self._reference_declination = config['reference_declination']
        self._pointings = config['pointings']
        self._bandwidth = config['channel_bandwidth']
        self._nbeams = config['nbeams']
        self._nants = nants
        self._nsubs = nsubs

        # Ignore AstropyWarning
        warnings.simplefilter('ignore', category=AstropyWarning)

        # Create AntennaArray object
        self._array = AntennaArray(array, self._reference_location[0], self._reference_location[1])

        # Calculate displacement vectors to each antenna from reference antenna
        # (longitude, latitude)
        self._reference_location = self._reference_location[0], self._reference_location[1]

        self._vectors_enu = np.full([self._nants, 3], np.nan)
        for i in range(self._nants):
            self._vectors_enu[i, :] = self._array.antenna_position(i)

        # Create initial weights
        self.weights = np.ones((self._nsubs, self._nbeams, self._nants), dtype=np.complex64)

        # Generate weights
        for beam in range(self._nbeams):
            self.point_array(beam, self._reference_declination, self._pointings[beam][0], self._pointings[beam][1])

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

            # Apply to weights
            self.weights[i, beam, :].real = real
            self.weights[i, beam, :].imag = imag

    def point_array(self, beam, ref_dec, ha, delta_dec):
        """ Calculate the phase shift between two antennas which is given by the phase constant (2 * pi / wavelength)
        multiplied by the projection of the baseline vector onto the plane wave arrival vector
        :param beam: Beam to which this applies
        :param right_ascension: Right ascension of source (astropy angle, or string that can be converted to angle)
        :param declination: Declination of source (astropy angle, or string that can be converted to angle)
        :param pointing_time: Time of observation (in format astropy time)
        :return: The phaseshift in radians for each antenna
        """

        # We must have a positive hour angle and non-zero
        if ha < 0:
            ha = Angle(ha + 360, u.deg)
        elif ha < 0.000001:
            ha = Angle(0.000001, u.deg)
        else:
            ha = Angle(ha, u.deg)

        # Type conversions if required
        ref_dec = Angle(ref_dec, u.deg)
        delta_dec = Angle(delta_dec, u.deg)

        # Convert RA DEC to ALT AZ
        alt, az = self._ha_dec_to_alt_az(ha, ref_dec + delta_dec, self._reference_location)

        # Point beam to required ALT AZ
        print("LAT: {}, HA: {}, DEC: {}, ALT: {}, AZ: {}".format(self._reference_location[1], ha.deg, ref_dec + delta_dec, alt.deg, az.deg))
        self.point_array_static(beam, alt, az)

    @staticmethod
    def _ha_dec_to_alt_az(hour_angle, declination, location):
        """ Calculate the altitude and azimuth coordinates of a sky object from right ascension and declination and time
        :param right_ascension: Right ascension of source (in astropy Angle on string which can be converted to Angle)
        :param declination: Declination of source (in astropy Angle on string which can be converted to Angle)
        :param time: Time of observation (as astropy Time")
        :param location: astropy EarthLocation
        :return: Array containing altitude and azimuth of source as astropy angle
        """
        lat = Angle(location[1], u.deg)

        alt = Angle(np.arcsin(np.sin(declination.rad) * np.sin(lat.rad) +
                              np.cos(declination.rad) * np.cos(lat.rad) * np.cos(hour_angle.rad)),
                    u.rad)

        # Condition where array is at zenith
        if abs(lat.deg - declination.deg) < 0.01:
            az = Angle(0, u.deg)
        else:
            az = Angle(np.arccos((np.sin(declination.rad) - np.sin(alt.rad) * np.sin(lat.rad)) /
                                 (np.cos(alt.rad) * np.cos(lat.rad))), u.rad)
            if np.sin(hour_angle.rad) >= 0:
                az = Angle(360 - az.deg, u.deg)

        return alt, az

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
    config['reference_antenna_location'] = [11.6459889, 44.52357778]
    config['reference_declination'] = 58.905452
    config['pointings'] = [[0, 0]]

    calib_coeffs = np.array([1+0j,
    0.74462+0.73813j,
    0.96279+0.16081j,
    1.219+0.134j,
    -0.32421+1.0584j,
    -0.15246+1.3064j,
    -1.0901+0.28728j,
    -1.1169+0.33127j,
    -0.61186-0.81568j,
    -0.16366-0.98011j,
    -0.54654-0.87314j,
    0.51988-0.84227j,
    0.87381+0.31297j,
    -0.67283-0.73793j,
    0.99691+0.29745j,
    0.69452+0.90883j,
    0.66416-0.77201j,
    0.89022+0.5477j,
    0.91526+0.32415j,
    1.0374+0.040547j,
    -0.532+0.9012j,
    -0.4766+0.87434j,
    -0.41321+0.94853j,
    -0.68958+0.9147j,
    -1.0694-0.46741j,
    0.26409-1.0113j,
    0.91304-0.62572j,
    0.50817-0.97052j,
    0.84344+0.41115j,
    0.51221-0.91149j,
    0.29735+0.93353j,
    0.20658+1.036j], dtype=np.complex64)

    pointing = Pointing(config, 1, 32)
    print pointing.weights

from numpy import ctypeslib
import numpy as np
import ctypes
import os

import datetime
import logging
from threading import Thread, Lock
import time

from astropy.coordinates import Angle, EarthLocation, SkyCoord, AltAz
from astropy.units import Quantity
from astropy import constants
from astropy import units as u
from astropy.time import Time

from pybirales.base import settings
from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule
from pybirales.blobs.beamformed_data import BeamformedBlob
from pybirales.blobs.dummy_data import DummyBlob

# Mute astropy warning
import warnings
from astropy.utils.exceptions import AstropyWarning

from pybirales.blobs.receiver_data import ReceiverBlob

warnings.simplefilter('ignore', category=AstropyWarning)


class Beamformer(ProcessingModule):
    """ Beamformer processing module """

    def __init__(self, config, input_blob=None):
        # This module needs an input blob of type channelised
        if type(input_blob) not in [DummyBlob, ReceiverBlob]:
            raise PipelineError("Beamformer: Invalid input data type, should be DummyBlob or ReceiverBlob")

        # Sanity checks on configuration
        if {'nbeams', 'antenna_locations', 'pointings', 'reference_pointing'} - set(config.settings()) != set():
            raise PipelineError("Beamformer: Missing keys on configuration "
                                "(nbeams, nants, antenna_locations, pointings)")
        self._nbeams = config.nbeams

        # Make sure that antenna locations is a list
        if type(config.antenna_locations) is not list:
            raise PipelineError("Beamformer: Expected list of antennas with long/lat/height as antenna locations")

        self._nthreads = 2
        if 'nthreads' in config.settings():
            self._nthreads = config.nthreads

        # Create placeholder for pointing class instance
        self._pointing = None

        # Wrap library containing C-implementation of beamformer
        self._beamformer = ctypes.CDLL("libbeamformer.so")
        complex_p = ctypeslib.ndpointer(np.complex64, ndim=1, flags='C')
        self._beamformer.beamform.argtypes = [complex_p, complex_p, complex_p, ctypes.c_uint32, ctypes.c_uint32,
                                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._beamformer.beamform.restype = None

        # Call superclass initialiser
        super(Beamformer, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Beamformer"

    def generate_output_blob(self):
        """ Generate output data blob """
        input_shape = dict(self._input.shape)
        datatype = self._input.datatype

        # Initialise pointing
        self._initialise(input_shape['nsubs'], input_shape['nants'])

        # Create output blob
        return BeamformedBlob(self._config, [('nbeams', self._nbeams),
                                             ('nsubs', input_shape['nsubs']),
                                             ('nsamp', input_shape['nsamp'])],
                              datatype=datatype)

    def _initialise(self, nsubs, nants):
        """ Initialise pointing """

        # Create pointing instance
        if self._pointing is None:
            self._pointing = Pointing(self._config, nsubs, nants)
            self._pointing.start()

    def process(self, obs_info, input_data, output_data):
        # Get data information
        nsamp = obs_info['nsamp']
        nsubs = obs_info['nsubs']
        nants = obs_info['nants']

        # If pointing is not initialise, initialise
        if self._pointing is None:
            self._initialise(nsubs, nants)

        # Get weights
        self._pointing.start_reading_weighs()

        # Apply pointing coefficients
        self._beamformer.beamform(input_data.ravel(), self._pointing.weights.ravel(), output_data.ravel(),
                                  nsamp, nsubs, self._nbeams, nants, self._nthreads)

        # Done using weights
        self._pointing.stop_reading_weights()

        # Update observation information
        obs_info['nbeams'] = self._nbeams
        logging.info("Beamformed data")


# ------------------------------------------------------------------------------------------------


class Pointing(Thread):
    """ Pointing class which periodically updates pointing weights """

    def __init__(self, config, nsubs, nants):

        # Call superclass initialiser
        super(Pointing, self).__init__()

        # Make sure that we have enough antenna locations
        if len(config.antenna_locations) != nants:
            raise PipelineError("Pointing: Mismatch between number of antennas and number of antenna locations")

        # Make sure that we have enough pointing
        if len(config.pointings) != config.nbeams:
            raise PipelineError("Pointing: Mismatch between number of beams and number of beam pointings")

        # Initialise Pointing
        array = config.antenna_locations
        self._start_center_frequency = settings.observation.start_center_frequency
        self._reference_pointing = config.reference_pointing
        self._pointings = config.pointings
        self._bandwidth = settings.observation.bandwidth
        self._nbeams = config.nbeams
        self._nants = nants
        self._nsubs = nsubs

        self._pointing_period = 5
        if 'pointing_period' in config.settings():
            self._pointing_period = config.pointing_period

        # Create AntennaArray object
        self._array = AntennaArray(array)

        # Calculate displacement vectors to each antenna from reference antenna
        self._reference_antenna_loc = EarthLocation.from_geodetic(self._array.lon, self._array.lat,
                                                                  height=self._array.height)
        self._baseline_vectors_ecef = self._calculate_ecef_baseline_vectors()
        self._baseline_vectors_enu = self._rotate_ecef_to_enu(self._baseline_vectors_ecef, self._array.lat,
                                                              self._array.lon)
        # Calculate East-North-Up vector
        self._vectors_enu = self._rotate_ecef_to_enu(self._baseline_vectors_ecef, self._array.lat, self._array.lon)

        # Lock to be used for synchronised access
        self._lock = Lock()

        # Create initial weights
        self.weights = np.zeros((self._nsubs, self._nbeams, self._nants), dtype=np.complex64)
        self._temp_weights = np.zeros((self._nsubs, self._nbeams, self._nants), dtype=np.complex64)

    def run(self):
        """ Run thread """
        while True:
            for beam in xrange(self._nbeams):
                self.point_array(beam, self._reference_pointing[0] + self._pointings[beam][0],
                                 self._reference_pointing[1] + self._pointings[beam][1])
            self._lock.acquire()
            self.weights = self._temp_weights.copy()
            self._lock.release()
            logging.info("Updated beamforming coefficients")
            time.sleep(self._pointing_period)

    def start_reading_weighs(self):
        """ Lock weights for access to beamformer """
        self._lock.acquire()

    def stop_reading_weights(self):
        """ Beamformer finished, unlock weights """
        self._lock.release()

    def point_array_static(self, beam, altitude, azimuth):
        """ Calculate the phase shift given the altitude and azimuth coordinates of a sky object as astropy angles
        :param beam: beam to which this applies
        :param altitude: altitude coordinates of a sky object as astropy angle
        :param azimuth: azimuth coordinates of a sky object as astropy angles
        :return: The phase shift in radians for each antenna
        """

        for i in xrange(self._nsubs):
            # Calculate complex coefficients
            frequency = Quantity(self._start_center_frequency + (i * self._bandwidth / self._nsubs), u.Hz)
            real, imag = self._phaseshifts_from_altitude_azimuth(altitude.rad, azimuth.rad, frequency, self._vectors_enu)

            # Apply to weights file
            self._temp_weights[i, beam, :].real = real
            self._temp_weights[i, beam, :].imag = imag

    def point_array(self, beam, right_ascension, declination, pointing_time=None):
        """ Calculate the phase shift between two antennas which is given by the phase constant (2 * pi / wavelength)
        multiplied by the projection of the baseline vector onto the plane wave arrival vector
        :param beam: Beam to which this applies
        :param right_ascension: Right ascension of source (astropy angle, or string that can be converted to angle)
        :param declination: Declination of source (astropy angle, or string that can be converted to angle)
        :param pointing_time: Time of observation (in format astropy time)
        :return: The phaseshift in radians for each antenna
        """

        # Type conversions if required
        if type(right_ascension) is not Angle:
            if type(right_ascension) is str:
                right_ascension = Angle(right_ascension)
            else:
                right_ascension = Angle(right_ascension, u.deg)

        if type(declination) is not Angle:
            if type(declination) is str:
                declination = Angle(declination)
            else:
                declination = Angle(declination, u.deg)

        if pointing_time is None:
            pointing_time = Time(datetime.datetime.utcnow(), scale='utc')

        alt, az = self._ra_dec_to_alt_az(Angle(right_ascension), Angle(declination),
                                         Time(pointing_time), self._reference_antenna_loc)
        self.point_array_static(beam, alt, az)

    def _calculate_ecef_baseline_vectors(self):
        """ Calculate the displacement vectors in metres for each antenna relative to master antenna.
        :return: an (n, 3) array of displacement vectors for each antenna
        """
        displacement_vectors = np.full([self._nants, 3], np.nan)
        array_positions = self._array.positions()

        ref_loc = EarthLocation.from_geodetic(lon=self._array.lon, lat=self._array.lat,
                                              height=self._array.height, ellipsoid='WGS84')
        ant_ref_pos = ref_loc.to_geocentric()
        ant_ref_values = np.array(
                [ant_ref_pos[0].to(u.m).value, ant_ref_pos[1].to(u.m).value, ant_ref_pos[2].to(u.m).value])

        for i in xrange(0, self._nants):
            ant_i_pos = (array_positions[0][i], array_positions[1][i], array_positions[2][i])
            ant_i_pos_values = np.array(
                    [ant_i_pos[0].to(u.m).value, ant_i_pos[1].to(u.m).value, ant_i_pos[2].to(u.m).value])
            displacement_vectors[i, :] = ant_i_pos_values - ant_ref_values

        return displacement_vectors

    def _rotate_ecef_to_enu(self, ecef_vectors, latitude, longitude):
        """
        Rotates a set of vectors expressed in the geocentric Earth-Centred Earth-Fixed coordinate system to the local
        East-North-Up surface tangential (horizontal) coordinate system with the specified latitude, longitude.
        :param ecef_vectors: (n, 3) array of vectors expressed in ECEF
        :param latitude: Geodectic latitude (astropy angle)
        :param longitude: Geodectic longitude (astropy angle)
        :return: array containing vectors rotated to East North Up coordinate system
        """

        rotation_matrix = self._rotation_matrix_ecef_to_enu(latitude, longitude)
        vectors_enu = np.dot(rotation_matrix, np.array(ecef_vectors).transpose())
        return vectors_enu.transpose()

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
        sky_coordinates = SkyCoord(ra=right_ascension, dec=declination)
        c_altaz = sky_coordinates.transform_to(AltAz(obstime=time, location=location))

        return c_altaz.alt, c_altaz.az

    @staticmethod
    def _rotation_matrix_ecef_to_enu(latitude, longitude):
        """
        Compute a rotation matrix for converting vectors in ECEF to local
        ENU (horizontal) cartesian coordinates at the specified latitude, longitude.
        :param latitude: The geodetic latitude of the reference point
        :param longitude: The geodetic longitude of the reference point
        :return: A 3*3 rotation (direction cosine) matrix.
        """
        sin_lat = np.sin(latitude)
        cos_lat = np.cos(latitude)
        sin_long = np.sin(longitude)
        cos_long = np.cos(longitude)

        rotation_matrix = np.zeros((3, 3), dtype=float)
        rotation_matrix[0, 0] = -sin_long
        rotation_matrix[0, 1] = cos_long
        rotation_matrix[0, 2] = 0

        rotation_matrix[1, 0] = -sin_lat * cos_long
        rotation_matrix[1, 1] = -sin_lat * sin_long
        rotation_matrix[1, 2] = cos_lat

        rotation_matrix[2, 0] = cos_lat * cos_long
        rotation_matrix[2, 1] = cos_lat * sin_long
        rotation_matrix[2, 2] = sin_lat

        return rotation_matrix

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

    def __init__(self, positions=None, filepath=None):
        self._ref_lat = None
        self._ref_lon = None
        self._ref_height = None

        self._x = None
        self._y = None
        self._z = None
        self._height = None

        self._n_antennas = None
        self._baseline_vectors_ecef = None
        self._baseline_vectors_enu = None

        # If filepath is defined, initialise array
        if filepath:
            self.load_from_file(filepath)

        # If positions is defined, initialise array
        if positions:
            self.load_from_positions(positions)

    def positions(self):
        """
        :return: Antenna positions
        """
        return self._x, self._y, self._z, self._height

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
        Retern reference height
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
        self._x = []
        self._y = []
        self._z = []
        self._height = []

        for i in xrange(len(positions)):
            latitude = Angle(str(positions[i][0]) + 'd')
            longitude = Angle(str(positions[i][1]) + 'd')
            height = u.Quantity(positions[i][2], u.m)

            loc = EarthLocation.from_geodetic(lon=longitude, lat=latitude, height=height, ellipsoid='WGS84')
            (x, y, z) = loc.to_geocentric()

            self._x.append(x)
            self._y.append(y)
            self._z.append(z)
            self._height.append(height)

        # Use first antenna as reference antenna
        self._ref_lat = Angle(str(positions[0][0]) + 'd')
        self._ref_lon = Angle(str(positions[0][1]) + 'd')
        self._ref_height = u.Quantity(positions[0][2], u.m)

    def load_from_file(self, filepath):
        """ Load antenna positions from file
        :param filepath: Path to file
        """

        # Check that file exists
        if not os.path.exists(filepath):
            logging.error("Could not load antenna positions from %s" % filepath)
            return

        # Load file
        array = np.loadtxt(filepath, delimiter=',')

        # Keep track of how many antennas are in the array
        self._n_antennas = len(array)

        # Convert file data to astropy format, and store lat, lon and height for each antenna
        if self._x is None:
            self._x = []

        if self._y is None:
            self._y = []

        if self._z is None:
            self._z = []

        if self._height is None:
            self._height = []

        for i in range(len(array)):
            latitude = Angle(str(array[i, 0]) + 'd')
            longitude = Angle(str(array[i, 1]) + 'd')
            height = u.Quantity(array[i, 2], u.m)

            loc = EarthLocation.from_geodetic(lon=longitude, lat=latitude, height=height, ellipsoid='WGS84')
            (x, y, z) = loc.to_geocentric()

            self._x.append(x)
            self._y.append(y)
            self._z.append(z)
            self._height.append(height)

        # Use first antenna as reference antenna
        self._ref_lat = Angle(str(array[0, 0]) + 'd')
        self._ref_lon = Angle(str(array[0, 1]) + 'd')
        self._ref_height = u.Quantity(array[0, 2], u.m)

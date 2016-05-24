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

# Required for IERS tables and what not
import astroplan


class Beamformer(ProcessingModule):
    """ Beamformer processing module """

    def __init__(self, config, input_blob=None):
        # This module needs an input blob of type channelised
        if type(input_blob) not in [DummyBlob, ReceiverBlob]:
            raise PipelineError("Beamformer: Invalid input data type, should be DummyBlob or ReceiverBlob")

        # Sanity checks on configuration
        if {'nbeams', 'antenna_locations', 'pointings', 'reference_pointing', 'reference_antenna_location'} \
                - set(config.settings()) != set():
            raise PipelineError("Beamformer: Missing keys on configuration "
                                "(nbeams, nants, antenna_locations, pointings)")
        self._nbeams = config.nbeams

        # Make sure that antenna locations is a list
        if type(config.antenna_locations) is not list:
            raise PipelineError("Beamformer: Expected list of antennas with long/lat/height as antenna locations")

        # Number of threads to use
        self._nthreads = 2
        if 'nthreads' in config.settings():
            self._nthreads = config.nthreads

        # Check if we have to move the telescope
        self._move_to_dec = False
        if 'move_to_dec' in config.settings():
            self._move_to_dec = config.move_to_dec

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
        obs_info['reference_pointing'] = self._config.reference_pointing
        obs_info['pointings'] = self._config.pointings
        logging.info("Beamformed data")


# ------------------------------------------------------------------------------------------------


class Pointing(Thread):
    """ Pointing class which periodically updates pointing weights """

    def __init__(self, config, nsubs, nants):

        # Call superclass initialiser
        super(Pointing, self).__init__()
        self._set_daemon()

        # Make sure that we have enough antenna locations
        if len(config.antenna_locations) != nants:
            raise PipelineError("Pointing: Mismatch between number of antennas and number of antenna locations")

        # Make sure that we have enough pointing
        if len(config.pointings) != config.nbeams:
            raise PipelineError("Pointing: Mismatch between number of beams and number of beam pointings")

        # Check if we're using static beams
        self._static_beams = False
        if "static_beams" in config.settings():
            self._static_beams = config.static_beams

        # Initialise Pointing
        array = config.antenna_locations
        self._start_center_frequency = settings.observation.start_center_frequency
        self._reference_location = config.reference_antenna_location
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
        self._array = AntennaArray(array, self._reference_location[0], self._reference_location[1])

        # Calculate displacement vectors to each antenna from reference antenna
        self._reference_location = EarthLocation.from_geodetic(self._array.lon, self._array.lat,
                                                               height=self._array.height)

        self._vectors_enu = np.full([self._nants, 3], np.nan)
        for i in range(self._nants):
            self._vectors_enu[i, :] = self._array.antenna_position(i)

        # Lock to be used for synchronised access
        self._lock = Lock()

        # Create initial weights
        self.weights = np.zeros((self._nsubs, self._nbeams, self._nants), dtype=np.complex64)
        self._temp_weights = np.zeros((self._nsubs, self._nbeams, self._nants), dtype=np.complex64)

    def run(self):
        """ Run thread """
        while True:

            # Get time once (so that all beams use the same time for pointing)
            pointing_time = Time(datetime.datetime.utcnow(), scale='utc', location=self._reference_location)

            for beam in xrange(self._nbeams):
                self.point_array(beam, self._reference_pointing[0], self._reference_pointing[1],
                                 self._pointings[beam][0], self._pointings[beam][1],
                                 pointing_time)

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
            frequency = Quantity(self._start_center_frequency + (i * self._bandwidth / self._nsubs), u.MHz)
            real, imag = self._phaseshifts_from_altitude_azimuth(altitude.rad, azimuth.rad, frequency, self._vectors_enu)

            # Apply to weights file
            self._temp_weights[i, beam, :].real = real
            self._temp_weights[i, beam, :].imag = imag

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
        if self._static_beams:
            ref_ra = Angle(pointing_time.sidereal_time('mean'))

        # Convert RA DEC to ALT AZ
        alt, az = self._ra_dec_to_alt_az(ref_ra + delta_ra,
                                         ref_dec + delta_dec,
                                         Time(pointing_time),
                                         self._reference_location)

        # Point beam to required ALT AZ
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
        sky_coordinates = SkyCoord(ra=right_ascension, dec=declination)
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
        self._x = [positions[i][0] for i in xrange(len(positions))]
        self._y = [positions[i][1] for i in xrange(len(positions))]
        self._z = [positions[i][2] for i in xrange(len(positions))]
        self._height = [0 for i in xrange(len(positions))]

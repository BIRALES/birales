import ctypes
import logging
import logging as log
import warnings

import numpy as np
from math import cos, sin, atan2, asin
from astropy import constants
from astropy import units as u
from astropy.coordinates import Angle
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyWarning
from numpy import ctypeslib

from pybirales import settings
from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.beamformed_data import BeamformedBlob
from pybirales.pipeline.blobs.dummy_data import DummyBlob
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob

# Mute Astropy Warnings
warnings.simplefilter('ignore', category=AstropyWarning)


class Beamformer(ProcessingModule):
    """ Beamformer processing module """

    def __init__(self, config, input_blob=None):
        # This module needs an input blob of type channelised
        self._validate_data_blob(input_blob, valid_blobs=[DummyBlob, ReceiverBlob])

        # Sanity checks on configuration
        if {'nbeams', 'antenna_locations', 'pointings', 'reference_antenna_location', 'reference_declination'} \
                - set(config.settings()) != set():
            raise PipelineError("Beamformer: Missing keys on configuration "
                                "(nbeams, nants, antenna_locations, pointings)")
        self._nbeams = config.nbeams

        self._disable_antennas = None
        if 'disable_antennas' in config.settings():
            self._disable_antennas = config.disable_antennas

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
                                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._beamformer.beamform.restype = None

        # Call superclass initializer
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
        return BeamformedBlob(self._config, [('npols', input_shape['npols']),
                                             ('nbeams', self._nbeams),
                                             ('nsubs', input_shape['nsubs']),
                                             ('nsamp', input_shape['nsamp'])],
                              datatype=datatype)

    def _initialise(self, nsubs, nants):
        """ Initialise pointing """

        # Create pointing instance
        if self._pointing is None:
            self._pointing = Pointing(self._config, nsubs, nants)
            if self._disable_antennas is not None:
                self._pointing.disable_antennas(self._disable_antennas)

    def process(self, obs_info, input_data, output_data):

        # Get data information
        nsamp = obs_info['nsamp']
        nsubs = obs_info['nsubs']
        nants = obs_info['nants']
        npols = obs_info['npols']

        # If pointing is not initialise, initialise
        if self._pointing is None:
            self._initialise(nsubs, nants)

        # Apply pointing coefficients
        self._beamformer.beamform(input_data.ravel(), self._pointing.weights.ravel(), output_data.ravel(),
                                  nsamp, nsubs, self._nbeams, nants, npols, self._nthreads)

        # Update observation information
        obs_info['nbeams'] = self._nbeams
        obs_info['pointings'] = self._config.pointings
        obs_info['beam_az_el'] = self._pointing.beam_az_el

        return obs_info

    def stop(self):
        logging.info('Stopping %s module', self.name)
        self._stop.set()


class Pointing(object):
    """ Pointing class which periodically updates pointing weights """

    def __init__(self, config, nsubs, nants):

        # Make sure that we have enough antenna locations
        if len(config.antenna_locations) != nants:
            logging.error("Pointing: Mismatch between number of antennas and number of antenna locations")

        # Make sure that we have enough pointing
        if len(config.pointings) != config.nbeams:
            logging.error("Pointing: Mismatch between number of beams and number of beam pointings")

        # Initialise Pointing
        array = config.antenna_locations
        self._start_center_frequency = settings.observation.start_center_frequency
        self._reference_location = config.reference_antenna_location
        self._reference_declination = config.reference_declination
        self._pointings = config.pointings
        self._bandwidth = settings.observation.channel_bandwidth
        self._nbeams = config.nbeams
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

        # Create placeholder for azimuth and elevation
        self.beam_az_el = np.zeros((self._nbeams, 2))

        # Generate weights
        for beam in range(self._nbeams):
            self.point_array(beam, self._reference_declination, self._pointings[beam][0], self._pointings[beam][1])

        # Ignore AstropyWarning
        warnings.simplefilter('ignore', category=AstropyWarning)

    def disable_antennas(self, antennas):
        """ Disable any antennas """
        for antenna in antennas:
            logging.info("Disabling antenna {}".format(antenna))
            self.weights[:, :, antenna] = np.zeros((self._nsubs, self._nbeams))

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

    def point_array_birales(self, beam, ref_dec, ha, delta_dec):
        """ Calculate the phase shift between two antennas which is given by the phase constant (2 * pi / wavelength)
        multiplied by the projection of the baseline vector onto the plane wave arrival vector
        :param beam: Beam to which this applies
        :param ref_dec: Reference declination (center of FoV)
        :param ha: Hour angel of source (astropy angle, or string that can be converted to angle)
        :param delta_dec: Declination of source (astropy angle, or string that can be converted to angle)
        :return: The phaseshift in radians for each antenna
        """
        # Type conversions if required
        ref_dec = Angle(ref_dec, u.deg)
        delta_dec = Angle(delta_dec, u.deg)

        # Convert RA DEC to ALT AZ
        primary_alt, primary_az = self._ha_dec_to_alt_az(Angle(0, u.deg), ref_dec, self._reference_location)

        # We must have a positive hour angle and non-zero
        if ha < 0:
            ha = Angle(ha + 360, u.deg)
        elif ha < 0.0001:
            ha = Angle(0.0001, u.deg)
        else:
            ha = Angle(ha, u.deg)

        # Unit position vector RX-sat in sensor reference frame
        rhou_sat_rx_sens_rf = np.matrix([cos(-ha.rad) * cos(delta_dec.rad),
                                        sin(-ha.rad) * cos(delta_dec.rad),
                                        sin(delta_dec.rad)])

        # Unit position vector RX-sat in NWZ reference frame
        alpha = -primary_az.rad

        rot1 = np.matrix([[cos(alpha), sin(alpha), 0],
                          [-sin(alpha), cos(alpha), 0],
                          [0, 0, 1]])

        phi = -primary_alt.rad
        rot2 = np.matrix([[cos(phi), 0, -sin(phi)],
                          [0, 1, 0],
                          [sin(phi), 0, cos(phi)]])

        rhou_sat_rx_nwz = (rot2 * rot1).T * rhou_sat_rx_sens_rf.T

        beam_az = Angle(-atan2(rhou_sat_rx_nwz[1], rhou_sat_rx_nwz[0]), u.rad)
        beam_el = Angle(asin(rhou_sat_rx_nwz[2]), u.rad)

        # Save beam azimuth and elevation
        self.beam_az_el[beam] = beam_az, beam_el

        # Point beam to required ALT AZ
        log.debug("LAT: {}, HA: {}, DEC: {}, ALT: {}, AZ: {}".format(self._reference_location[1], ha.deg, ref_dec +
                                                                     delta_dec, beam_el.deg, beam_az.deg))
        self.point_array_static(beam, beam_el, beam_az)

    def point_array(self, beam, ref_dec, ha, delta_dec):
        """ Calculate the phase shift between two antennas which is given by the phase constant (2 * pi / wavelength)
        multiplied by the projection of the baseline vector onto the plane wave arrival vector
        :param beam: Beam to which this applies
        :param right_ascension: Right ascension of source (astropy angle, or string that can be converted to angle)
        :param declination: Declination of source (astropy angle, or string that can be converted to angle)
        :param pointing_time: Time of observation (in format astropy time)
        :return: The phaseshift in radians for each antenna
        """
        # Type conversions if required
        ref_dec = Angle(ref_dec, u.deg)
        delta_dec = Angle(delta_dec, u.deg)
        dec = ref_dec + delta_dec

        # Declination depends on DEC divide by DEC
        ha = ha / np.cos(dec.rad)

        # We must have a positive hour angle and non-zero
        if ha < 0:
            ha = Angle(ha + 360, u.deg)
        elif ha < 0.0001:
            ha = Angle(0.0001, u.deg)
        else:
            ha = Angle(ha, u.deg)

        # Convert RA DEC to ALT AZ
        alt, az = self._ha_dec_to_alt_az(ha, dec, self._reference_location)

        # Point beam to required ALT AZ
        log.debug("LAT: {}, HA: {}, DEC: {}, ALT: {}, AZ: {}".format(self._reference_location[1], ha.deg, dec, alt.deg, az.deg))
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
            [np.sin(azimuth) * np.cos(altitude), np.cos(azimuth) * np.cos(altitude), np.sin(altitude)])

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

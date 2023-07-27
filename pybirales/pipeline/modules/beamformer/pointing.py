import datetime
import logging
import logging as log
import warnings
from math import cos, sin, atan2, asin

import numpy as np
from astropy import constants
from astropy import units as u
from astropy.coordinates import Angle
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyWarning

from pybirales import settings
from pybirales.repository.models import CalibrationObservation, Observation
from pybirales.pipeline.base.definitions import InvalidCalibrationCoefficientsException, PipelineError

# Mute Astropy Warnings
warnings.simplefilter('ignore', category=AstropyWarning)


class Pointing:
    """ Pointing class which periodically updates pointing weights """

    ANTENNA_LOCATION_DTYPE = np.dtype([('group', 'U4'), ('local_id', 'i2'), ('global_id', 'i2'),
                                       ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('height', 'f4')])

    def __init__(self, config, nof_channels, nof_antennas):
        """ Class initialiser
        @param config: Pointing configuration
        @param config: Number of frequency channels"""

        # Load settings
        self._start_center_frequency = settings.observation.start_center_frequency
        self._reference_location = settings.instrument.reference_antenna_location
        self._reference_declinations = config.reference_declinations
        self._bandwidth = settings.observation.channel_bandwidth
        self._nof_beams_per_subarray = config.nof_beams_per_subarray
        self._nof_subarrays = config.nof_subarrays
        self._nof_antennas = nof_antennas
        self._nof_channels = nof_channels

        # Sanity check on number of declinations
        if self._nof_subarrays != len(self._reference_declinations):
            raise PipelineError(f"Pointing. Number of reference declinations ({len(self._reference_declinations)}) "
                                f"!= number of subarrays {self._nof_subarrays}")

        # Load antenna locations
        self._antenna_locations = self.load_antenna_locations()

        # Load required pointing map
        self._pointings = self.load_beam_pointings()

        # Compute total number of beams and antenna per subarray
        self._nof_beams = len(self._pointings) * self._nof_subarrays
        self._nof_antenna_per_subarray = self._nof_antennas // self._nof_subarrays

        # Load calibration coefficients
        self._calibration_coefficients = self.load_calibration_coefficients()

        # Calculate displacement vectors to each antenna from reference antenna (longitude, latitude)
        self._reference_location = self._reference_location[0], self._reference_location[1]

        # Create initial weights. Use zeros instead of unit to easily determine whether an antenna should be used
        # for a given beam
        self._pointing_weights = np.zeros((self._nof_channels, self._nof_beams, self._nof_antennas), dtype=np.complex64)

        # Create placeholder for azimuth and elevation
        self._beam_azimuth_elevation = np.zeros((self._nof_beams, 2))

        # Loop over subarrays and beams and generate pointing weight for each
        global_beam_index = 0
        for subarray_index in range(self._nof_subarrays):
            for beam_index, (hour_angle, delta_declination) in enumerate(self._pointings):
                self.point_array_birales(global_beam_index, self._reference_declinations[subarray_index],
                                         hour_angle, delta_declination,
                                         subarray_index * self._nof_antenna_per_subarray,
                                         (subarray_index + 1) * self._nof_antenna_per_subarray)
                global_beam_index += 1

        log.info("Generated beam pointings")

    @property
    def pointing_weights(self):
        return self._pointing_weights

    def load_antenna_locations(self):
        """ Load and interpret antenna locations from config """

        # Alias for shorter code
        locations = settings.instrument.antenna_locations

        # If antenna locations is a list, convert to dictionary
        if type(locations) is list:
            locations = {'N1': locations}

        # Get total number of antenna entries
        nof_antennas = sum([len(v) for _, v in locations.items()])

        # Make sure that we have enough antenna locations
        if nof_antennas != self._nof_antennas:
            raise PipelineError("Pointing: Mismatch between number of antennas and number of antenna locations")

        # Create numpy structure
        antenna_locations = np.zeros(nof_antennas, dtype=self.ANTENNA_LOCATION_DTYPE)

        # Antenna locations will be a dictionary, with each group of 32 antennas assigned to a different
        # entry in the dict. Go through the
        global_index = 0
        for group in sorted(locations.keys()):
            for i, antenna in enumerate(locations[group]):
                antenna_locations['group'][global_index] = group
                antenna_locations['local_id'][global_index] = i
                antenna_locations['global_id'][global_index] = global_index
                antenna_locations['x'][global_index] = antenna[0]
                antenna_locations['y'][global_index] = antenna[1]
                antenna_locations['z'][global_index] = antenna[2]
                antenna_locations['height'][global_index] = 0
                global_index += 1

        return antenna_locations

    def load_beam_pointings(self):
        """ Load beam pointings from file """
        # Sanity check on number of beams per subarray and number of subarrays
        if self._nof_subarrays == 1 and self._nof_beams_per_subarray not in [1, 32, 64, 128, 256] or \
                self._nof_subarrays == 2 and self._nof_beams_per_subarray not in [1, 32, 64, 128] or \
                self._nof_subarrays == 4 and self._nof_beams_per_subarray not in [1, 32, 64] or \
                self._nof_subarrays == 8 and self._nof_beams_per_subarray not in [1, 32]:
            raise PipelineError(f"Pointing: Unsupported combination of number of subarrays {self._nof_subarrays} "
                                f"and number of beams per subarray {self._nof_beams_per_subarray}")

        # Sanity check on number of subarrays and number of antennas
        if self._nof_subarrays == 1 and self._nof_antennas not in [32, 64, 128, 256] \
                or self._nof_subarrays == 2 and self._nof_antennas not in [64, 128, 256] \
                or self._nof_subarrays == 4 and self._nof_antennas not in [128, 256] \
                or self._nof_subarrays == 8 and self._nof_antennas != 256:
            raise PipelineError(f"Pointing. Unsupported combination of number of antennas {self._nof_antennas} "
                                f"and number of subarrays {self._nof_subarrays}")

        # Return pointing map
        if self._nof_beams_per_subarray == 1:
            pointings = settings.beamformer.pointings_1_beam
        elif self._nof_beams_per_subarray == 32:
            pointings = settings.beamformer.pointings_32_beams
        elif self._nof_beams_per_subarray == 64:
            pointings = settings.beamformer.pointings_64_beams
        elif self._nof_beams_per_subarray == 128:
            pointings = settings.beamformer.pointings_128_beams
        elif self._nof_beams_per_subarray == 256:
            pointings = settings.beamformer.pointings_256_beams
        else:
            raise PipelineError(f"Pointing: Unsupported number of beams per subarray {self._nof_beams_per_subarray}")

        return pointings

    def load_calibration_coefficients(self):
        """ Load calibration coefficients from file or from database """

        # Initialise unit coefficients
        calibration_coefficients = np.ones(self._nof_antennas, dtype=np.complex64)

        try:
            if settings.beamformer.calibrate_subarrays:
                if "calibration_coefficients_filepath" in settings.beamformer.__dict__.keys() and \
                        settings.beamformer.calibration_coefficients_filepath != 'None':
                    logging.info(
                        f"Using calibration coefficients from {settings.beamformer.calibration_coefficients_filepath}")
                    return np.loadtxt(settings.beamformer.calibration_coefficients_filepath,
                                      dtype=complex)
                else:
                    return self._get_latest_calibration_coefficients()
            else:
                log.warning('No calibration coefficients applied to this observation')
        except InvalidCalibrationCoefficientsException as e:
            log.warning("Could not load coefficients from TCPO directory. Reason: {}".format(e))

        return calibration_coefficients

    def disable_antennas(self, antennas):
        """ Disable any antennas """
        for antenna in antennas:
            logging.info("Disabling antenna {}".format(antenna))
            self._pointing_weights[:, :, antenna] = np.zeros((self._nof_channels, self._nof_beams))

    def point_array_static(self, beam, altitude, azimuth, start_antenna_index, stop_antenna_index):
        """ Calculate the phase shift given the altitude and azimuth coordinates of a sky object as astropy angles
        :param beam: beam to which this applies
        :param altitude: altitude coordinates of a sky object as astropy angle
        :param azimuth: azimuth coordinates of a sky object as astropy angles
        :param start_antenna_index: Index of first antenna in beam
        :param stop_antenna_index: Index of last antenna in beam
        :return: The phase shift in radians for each antenna
        """

        # Aliases for start_antenna_index and stop_antenna_index
        start = start_antenna_index
        stop = stop_antenna_index

        # Extract required antennas and their positions
        antennas = self._antenna_locations[start: stop]
        location_vector = np.array([antennas['x'], antennas['y'], antennas['z']]).T

        for i in range(self._nof_channels):
            # Calculate complex coefficients
            frequency = Quantity(self._start_center_frequency + (i * self._bandwidth / self._nof_channels), u.MHz)
            real, imag = self._phaseshifts_from_altitude_azimuth(altitude.rad, azimuth.rad, frequency,
                                                                 location_vector)

            # Apply to weights
            self._pointing_weights[i, beam, start:stop].real = real
            self._pointing_weights[i, beam, start:stop].imag = -imag

            # Multiply generated weights with calibration coefficients
            self._pointing_weights[i, beam, start:stop] *= self._calibration_coefficients[start:stop]

    def point_array_birales(self, beam, reference_declination, hour_angle, delta_declination,
                            start_antenna_index, stop_antenna_index):
        """ Calculate the phase shift between two antennas which is given by the phase constant (2 * pi / wavelength)
        multiplied by the projection of the baseline vector onto the plane wave arrival vector
        :param beam: Beam to which this applies
        :param reference_declination: Reference declination (center of FoV)
        :param hour_angle: Hour angel of source (astropy angle, or string that can be converted to angle)
        :param delta_declination: Declination of source (astropy angle, or string that can be converted to angle)
        :param start_antenna_index: Index of first antenna in beam
        :param stop_antenna_index: Index of last antenna in beam
        :return: The phase shift in radians for each antenna
        """
        # Type conversions if required
        reference_declination = Angle(reference_declination, u.deg)
        delta_declination = Angle(delta_declination, u.deg)

        # Convert RA DEC to ALT AZ
        primary_alt, primary_az = self._ha_dec_to_alt_az(Angle(0, u.deg), reference_declination,
                                                         self._reference_location)

        # We must have a positive hour angle and non-zero
        if hour_angle < 0:
            hour_angle = Angle(hour_angle + 360, u.deg)
        elif hour_angle < 0.0001:
            hour_angle = Angle(0.0001, u.deg)
        else:
            hour_angle = Angle(hour_angle, u.deg)

        # Unit position vector RX-sat in sensor reference frame
        rhou_sat_rx_sens_rf = np.matrix([cos(-hour_angle.rad) * cos(delta_declination.rad),
                                         sin(-hour_angle.rad) * cos(delta_declination.rad),
                                         sin(delta_declination.rad)])

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

        beam_azimuth = Angle(-atan2(rhou_sat_rx_nwz[1], rhou_sat_rx_nwz[0]), u.rad)
        beam_elevation = Angle(asin(rhou_sat_rx_nwz[2]), u.rad)

        # Save beam azimuth and elevation
        self._beam_azimuth_elevation[beam] = beam_azimuth.deg, beam_elevation.deg

        # Generate antenna weights for beam
        self.point_array_static(beam, beam_elevation, beam_azimuth, start_antenna_index, stop_antenna_index)

        log.debug("LAT: {}, HA: {}, DEC: {}, EL: {}, AZ: {}".format(self._reference_location[1], hour_angle.deg,
                                                                    reference_declination + delta_declination,
                                                                    beam_elevation.deg, beam_azimuth.deg))

    def point_array(self, beam, reference_declination, hour_angle, delta_declination,
                    start_antenna_index, stop_antenna_index):
        """ NOTE: Deprecated, superseded by point_array_birales
        Calculate the phase shift between two antennas which is given by the phase constant (2 * pi / wavelength)
        multiplied by the projection of the baseline vector onto the plane wave arrival vector
        :param beam: Beam to which this applies
        :param reference_declination: Reference declination (center of FoV)
        :param hour_angle: Hour angel of source (astropy angle, or string that can be converted to angle)
        :param delta_declination: Declination of source (astropy angle, or string that can be converted to angle)
        :param start_antenna_index: Index of first antenna in beam
        :param stop_antenna_index: Index of last antenna in beam
        :return: The phase shift in radians for each antenna
        """
        # Type conversions if required
        reference_declination = Angle(reference_declination, u.deg)
        delta_declination = Angle(delta_declination, u.deg)
        dec = reference_declination + delta_declination

        # Declination depends on DEC divide by DEC
        hour_angle = hour_angle / np.cos(dec.rad)

        # We must have a positive hour angle and non-zero
        if hour_angle < 0:
            hour_angle = Angle(hour_angle + 360, u.deg)
        elif hour_angle < 0.0001:
            hour_angle = Angle(0.0001, u.deg)
        else:
            hour_angle = Angle(hour_angle, u.deg)

        # Convert RA DEC to ALT AZ
        altitude, azimuth = self._ha_dec_to_alt_az(hour_angle, dec, self._reference_location)

        # Point beam to required ALT AZ
        self.point_array_static(beam, altitude, azimuth, start_antenna_index, stop_antenna_index)

        log.debug("LAT: {}, HA: {}, DEC: {}, ALT: {}, AZ: {}".format(self._reference_location[1], hour_angle.deg, dec,
                                                                     altitude.deg, azimuth.deg))

    @staticmethod
    def _ha_dec_to_alt_az(hour_angle, declination, location):
        """ Calculate the altitude and azimuth coordinates of a sky object from right ascension and declination and time
        :param hour_angle: Hour angel of source (astropy angle, or string that can be converted to angle)
        :param declination: Declination of source (in astropy Angle on string which can be converted to Angle)
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
            temp = (np.sin(declination.rad) - np.sin(alt.rad) * np.sin(lat.rad)) / (np.cos(alt.rad) * np.cos(lat.rad))
            if temp < 0 and (temp + 1) < 0.0001:
                temp = -1
            elif temp > 0 and (temp - 1) < 0.0001:
                temp = 1
            az = Angle(np.arccos(temp), u.rad)
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

    def _get_latest_calibration_coefficients(self):
        """
        Read the calibration coefficients from file.
        :return:
        """
        # If observation id is false, use current time
        if not settings.observation.id:
            observation_start = datetime.datetime.utcnow()
            observation = None
        else:
            observation = Observation.objects.get(id=settings.observation.id)
            observation_start = observation.principal_created_at

        # Find the calibration algorithm whose principal start time is closest to the observation principal start time
        calib_obs = CalibrationObservation.objects(principal_created_at__lte=observation_start, status="finished").order_by(
            '-principal_created_at', 'created_at').first()

        if len(calib_obs) < 1:
            calib_obs = CalibrationObservation.objects(created_at__lte=observation_start, status="finished").order_by(
                'created_at').first()

        if len(calib_obs) < 1:
            raise InvalidCalibrationCoefficientsException("No suitable calibration coefficients files were found")

        log.info(
            'Selected Calibration obs: {}. Created At: {:%d-%m-%Y %H:%M:%S}. Principal: {:%d-%m-%Y %H:%M:%S}.'.format(
                calib_obs.name, calib_obs.created_at, calib_obs.principal_created_at))

        calibration_coefficients = np.array(calib_obs.real) + np.array(calib_obs.imag) * 1j

        if not len(calibration_coefficients) == self._nof_antennas:
            raise InvalidCalibrationCoefficientsException(
                "Number of calibration coefficients does not match number of antennas")

        if not np.iscomplexobj(calibration_coefficients):
            raise InvalidCalibrationCoefficientsException("Calibration coefficients type is not complex")

        log.info('Calibration coefficients loaded successfully')

        if observation and settings.observation.id:
            observation.calibration_observation = calib_obs.id
            observation.save()

        return calibration_coefficients

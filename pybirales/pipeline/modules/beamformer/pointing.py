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
from pybirales.pipeline.modules.beamformer.antenna_array import AntennaArray
from pybirales.repository.models import CalibrationObservation, Observation
from pybirales.pipeline.base.definitions import InvalidCalibrationCoefficientsException, PipelineError

# Mute Astropy Warnings
warnings.simplefilter('ignore', category=AstropyWarning)


class Pointing:
    """ Pointing class which periodically updates pointing weights """

    def __init__(self, config, nsubs, nants):

        # Make sure that we have enough antenna locations
        if len(settings.instrument.antenna_locations) != nants:
            raise PipelineError("Pointing: Mismatch between number of antennas and number of antenna locations")

        # Make sure that we have enough pointing
        if len(config.pointings) != config.nbeams:
            raise PipelineError("Pointing: Mismatch between number of beams and number of beam pointings")

        # Load settings
        array = settings.instrument.antenna_locations
        self._start_center_frequency = settings.observation.start_center_frequency
        self._reference_location = settings.instrument.reference_antenna_location
        self._reference_declination = config.reference_declination

        self._bandwidth = settings.observation.channel_bandwidth
        self._nbeams = config.nbeams
        self._nants = nants
        self._nsubs = nsubs

        # Initialise pointings
        self._pointings = config.pointings

        log.info('Reference declination: {}'.format(self._reference_declination))

        # Calibration coefficients
        self._calib_coeffs = np.ones(self._nants, dtype=np.complex64)

        try:
            if settings.beamformer.apply_calib_coeffs:
                if "calibration_coefficients_filepath" in settings.beamformer.__dict__.keys() and \
                        settings.beamformer.calibration_coefficients_filepath != 'None':
                    self._calib_coeffs = np.loadtxt(settings.beamformer.calibration_coefficients_filepath,
                                                    dtype=complex)
                    logging.info(
                        f"Using calibration coefficients from {settings.beamformer.calibration_coefficients_filepath}")
                else:
                    self._calib_coeffs = self._get_latest_calib_coeffs()
            else:
                log.warning('No calibration coefficients applied to this observation')
        except InvalidCalibrationCoefficientsException as e:
            log.warning("Could not load coefficients from TCPO directory. Reason: {}".format(e))

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
            self.point_array_birales(beam, self._reference_declination, self._pointings[beam][0],
                                     self._pointings[beam][1])

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
            # print frequency, self._nsubs
            # Apply to weights
            self.weights[i, beam, :].real = real
            self.weights[i, beam, :].imag = -imag

            # Multiply generated weights with calibration coefficients
            self.weights[i, beam, :] *= self._calib_coeffs

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
        self.beam_az_el[beam] = beam_az.deg, beam_el.deg

        # Point beam to required ALT AZ
        log.debug("LAT: {}, HA: {}, DEC: {}, EL: {}, AZ: {}".format(self._reference_location[1], ha.deg, ref_dec +
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
        log.debug("LAT: {}, HA: {}, DEC: {}, ALT: {}, AZ: {}".format(self._reference_location[1], ha.deg, dec, alt.deg,
                                                                     az.deg))
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

    def _get_latest_calib_coeffs(self):
        """
        Read the calibration coefficients from file.
        :return:
        """
        # If observation id is false, use current time
        if not settings.observation.id:
            obs_start = datetime.datetime.utcnow()
        else:
            obs = Observation.objects.get(id=settings.observation.id)
            obs_start = obs.principal_created_at

        # Find the calibration algorithm whose principal start time is closest to this observation's principal start time
        calib_obs = CalibrationObservation.objects(principal_created_at__lte=obs_start, status="finished").order_by(
            '-principal_created_at', 'created_at').first()

        if len(calib_obs) < 1:
            calib_obs = CalibrationObservation.objects(created_at__lte=obs_start, status="finished").order_by(
                'created_at').first()

        if len(calib_obs) < 1:
            raise InvalidCalibrationCoefficientsException("No suitable calibration coefficients files were found")

        log.info(
            'Selected Calibration obs: {}. Created At: {:%d-%m-%Y %H:%M:%S}. Principal: {:%d-%m-%Y %H:%M:%S}.'.format(
                calib_obs.name, calib_obs.created_at, calib_obs.principal_created_at))

        calib_coeffs = np.array(calib_obs.real) + np.array(calib_obs.imag) * 1j

        if not len(calib_coeffs) == self._nants:
            raise InvalidCalibrationCoefficientsException(
                "Number of calibration coefficients does not match number of antennas")

        if not np.iscomplexobj(calib_coeffs):
            raise InvalidCalibrationCoefficientsException("Calibration coefficients type is not complex")

        log.info('Calibration coefficients loaded successfully')

        if settings.observation.id:
            obs.calibration_observation = calib_obs.id
            obs.save()

        return calib_coeffs

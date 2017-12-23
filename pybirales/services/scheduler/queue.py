import datetime
import pytz
import logging as log
from pybirales.services.scheduler.exceptions import NoObservationsQueuedException, ObservationScheduledInPastException, \
    ObservationsConflictException
from pybirales.services.scheduler.observation import ScheduledCalibrationObservation
from pybirales.utilities.source_transit import get_best_calibration_obs, get_calibration_sources, \
    get_calibration_source_declination, get_next_tranist


class ObservationsQueue:
    def __init__(self):
        self._queue = []

        # Maximum amount of time allowed before the instrument needs to be re-calibrated (in hours)
        self._recalibration_time = datetime.timedelta(hours=24)

    def __iter__(self):
        for q in self._queue:
            yield q

    def _observation_conflict(self, observation):
        """
        Check that the passed on observation does not conflict with the queued observations

        :param observation: The observation to be scheduled
        :type observation: ScheduledObservation
        :raises ObservationsConflictException: The observation conflicts with the queued observations
        :return: None
        """

        start_time = observation.start_time_padded
        end_time = observation.end_time_padded

        for scheduled_observation in self._queue:
            so_start = scheduled_observation.start_time_padded
            so_end = scheduled_observation.end_time_padded

            # Check for time ranges where duration is not defined:
            if not (so_end and end_time):
                # If there is an observation that has no end and starts before this observation
                if not so_end and so_start < start_time:
                    raise ObservationsConflictException(observation, scheduled_observation)

                # If this observation starts before and does not end
                if not end_time and start_time < so_start:
                    raise ObservationsConflictException(observation, scheduled_observation)

            # If this observation overlaps another time range
            elif (start_time <= so_end) and (so_start <= end_time):
                raise ObservationsConflictException(observation, scheduled_observation)
        else:
            # No overlap detected
            log.info('No overlap detected between scheduled observations')

    def _add_observation(self, observation):
        """
        Schedule an observation

        :param observation: The observation to be scheduled
        :type observation: ScheduledObservation
        :return:
        """

        try:
            # Check if observation is in the future
            self._is_future_observation(observation)

            # Check if this observation conflicts with the scheduled observation
            self._observation_conflict(observation)

        except ObservationsConflictException:
            log.error("Observation '{}', could not be scheduled".format(observation.name))
        except ObservationScheduledInPastException:
            log.error("Observation '{}' could not be scheduled".format(observation.name))
        else:
            # Add the observation to the queue
            self._queue.append(observation)
            log.info("Observation '{}', queued successfully".format(observation.name))

    def _add_observations(self, observations):
        """
        Add observations to the queue. Once observations are added, schedule the calibration routines

        :param observations:
        :return:
        """

        for observation in observations:
            self._add_observation(observation)

        log.info('%s observations queued.', len(self._queue))

        log.info('Adding calibration routines for observation')
        for scheduled_observation in self._queue:
            self._calibrate_for(scheduled_observation)

        log.info('%s calibration routines added. Queue is ready.', len(5))

    @staticmethod
    def _is_future_observation(observation):
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        wait_seconds = (observation.start_time_padded - now).total_seconds()

        if wait_seconds < 1:
            raise ObservationScheduledInPastException(observation.parameters, observation.start_time_padded)

    def _calibrate_for(self, observation):
        """
        Return the calibration routine that is closest to the observation

        :param observation:
        :return:
        """

        # Get the scheduled calibration routines
        calibrations = [c for c in self._queue if isinstance(c, ScheduledCalibrationObservation)]

        # The closest calibration routine to the observation
        closest_calibration = None

        # Default minimum time
        min_time_delta = -1
        for c in calibrations:
            # Time between observation and calibration routine
            time_delta = (observation.start_time_padded - c.end_time).total_seconds()

            # Check that time delta is the smallest, and that it is less than the minimum re-calibration time
            if time_delta < min_time_delta and time_delta < self._recalibration_time.total_seconds():
                min_time_delta = time_delta
                closest_calibration = c

        # No calibration for this observation found - or its too early for this observation
        if not closest_calibration:
            # Find the next available slot to schedule a calibration
            closest_calibration = self._schedule_calibration_routine_for(observation)

            # Call the function recursively. Closest calibration should not be None for this observation
            self._calibrate_for(observation)

        return closest_calibration

    def _schedule_calibration_routine_for(self, closest_observation):
        calibration_obs = get_best_calibration_obs(closest_observation)

        # Create a calibration object for the closest_observation observation
        name = '{}_{}.calibration'.format(calibration_obs['name'], calibration_obs['date'])
        config_file = closest_observation.config_file
        dec = calibration_obs['parameters']['dec']
        start_time = calibration_obs['transit_time']
        start_time_padding = 0
        calibration_observation = ScheduledCalibrationObservation(name, config_file, dec, start_time, start_time_padding)

        try:
            # Check if this observation conflicts with the scheduled observation
            self._observation_conflict(calibration_observation)
        except ObservationsConflictException as exception:
            # change the date and try again
            pass
        except BaseException:
            pass

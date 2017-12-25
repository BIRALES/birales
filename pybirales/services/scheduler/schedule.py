import datetime
import logging as log
from pybirales.services.scheduler.exceptions import ObservationsConflictException
from pybirales.services.scheduler.observation import ScheduledCalibrationObservation, ScheduledObservation
from pybirales.utilities.source_transit import get_best_calibration_obs


class Schedule:
    def __init__(self, time_to_calibrate, recalibration_time):
        self._head = None
        self._tail = None

        self.n_calibrations = 0
        self.n_observations = 0

        # Maximum amount of time allowed before the instrument needs to be re-calibrated (in hours)
        self._recalibration_time = recalibration_time

        self._time_to_calibrate = time_to_calibrate

    def __len__(self):
        return self.n_calibrations + self.n_observations

    def is_empty(self):
        """
        Check if schedule is empty
        :return:
        """
        return len(self) == 0

    def _increment(self, obs):
        """
        Increment the size of the schedule by 1
        :param obs:
        :return:
        """
        if isinstance(obs, ScheduledCalibrationObservation):
            self.n_calibrations += 1
        elif isinstance(obs, ScheduledObservation):
            self.n_observations += 1

    def _conflicts(self, scheduled_observation, new_obs):
        """
        Check that the passed on observation does not conflict with the queued observations

        :param new_obs: The observation to be scheduled
        :type new_obs: ScheduledObservation
        :raises ObservationsConflictException: The observation conflicts with the queued observations
        :return: None
        """

        if scheduled_observation is None:
            log.info('Observation `{}` does not overlap with the scheduled observations')
            return False

        start_time = new_obs.start_time_padded
        end_time = new_obs.end_time_padded

        so_start = scheduled_observation.start_time_padded
        so_end = scheduled_observation.end_time_padded

        # Check for time ranges where duration is not defined:
        if not (so_end and end_time):
            # If there is an observation that has no end and starts before this observation
            if not so_end and so_start < start_time:
                raise ObservationsConflictException(new_obs, scheduled_observation)

            # If this observation starts before and does not end
            if not end_time and start_time < so_start:
                raise ObservationsConflictException(new_obs, scheduled_observation)

        # If this observation overlaps another time range
        elif (start_time <= so_end) and (so_start <= end_time):
            raise ObservationsConflictException(new_obs, scheduled_observation)

        log.debug("Observation `{}` is not in conflict with `{}`".format(new_obs.name, scheduled_observation.name))

        log.debug("Checking if `{}` is in conflict with `{}`".format(new_obs.name, scheduled_observation.name))
        return self._conflicts(scheduled_observation.next_observation, new_obs)

    def add_observation(self, new_obs):
        """

        :param new_obs:
        :return:
        """

        # Check if observation is in the future
        new_obs.is_in_future()

        # Check if this observation conflicts with the scheduled observation - todo
        self._conflicts(self._head, new_obs)

        if self._head is None:
            # First observation must always be a calibration observation
            self._head = self._find_calibration_source_for(new_obs)
            self._tail = new_obs

            self._head.next_observation = self._tail
            self._tail.prev_observation = self._head

            self._increment(self._head)
            self._increment(self._tail)

            return new_obs

        calibration_obs = self._calibrate_for(new_obs)
        if calibration_obs:
            calibration_obs.prev_observation = self._tail
            calibration_obs.next_observation = None

            self._tail.next_observation = calibration_obs
            self._tail.next = calibration_obs

            self._increment(calibration_obs)

            return self.add_observation(new_obs)

        new_obs.prev_observation = self._tail
        new_obs.next_observation = None

        self._tail.next_observation = new_obs
        self._tail.next = new_obs

        self._increment(new_obs)

        return new_obs

    def _calibrate_for(self, observation):
        """
        Return the calibration routine that is closest to the observation

        :param observation:
        :return:
        """

        closest_calibration = None
        current_obs = self._head
        min_time_delta = -1
        while current_obs is not None:
            if not isinstance(current_obs, ScheduledCalibrationObservation):
                current_obs = current_obs.next_observation
                continue

            # Time between observation and calibration routine
            time_delta = (observation.start_time_padded - current_obs.end_time).total_seconds()

            # Check that time delta is the smallest, and that it is less than the minimum re-calibration time
            if time_delta < min_time_delta and time_delta < self._recalibration_time.total_seconds():
                min_time_delta = time_delta
                closest_calibration = current_obs

            current_obs = current_obs.next_observation

        # No calibration for this observation found - or its too early for this observation
        if not closest_calibration:
            # Find the next available slot to schedule a calibration
            return self._find_calibration_source_for(observation)

        return closest_calibration

    def _find_calibration_source_for(self, observation):
        """
        Find an optimal calibration date for an observation.

        :param observation: The observation for which we want a calibration routine
        :return:
        """

        # Get the available calibration sources on this day (with a time delta)
        from_time = None
        if observation.prev_observation:
            from_time = observation.prev_observation.end_time_padded
        available_sources = get_best_calibration_obs(from_time, observation.start_time_padded, self._time_to_calibrate)

        max_flux = -1
        best_obs = None
        for source in available_sources:
            # Create a calibration object for the closest_observation observation
            calibration_observation = ScheduledCalibrationObservation(source, observation.config_file,
                                                                      prv_obs=observation)

            try:
                # Check if this observation conflicts with the scheduled observation
                self._conflicts(self._head, calibration_observation)

                if source['flux'] > max_flux:
                    best_obs = calibration_observation
            except ObservationsConflictException:
                log.warning("Calibration observation '{}' could not be scheduled".format(calibration_observation.name))

        return best_obs

    def __iter__(self):
        """
        Iterate over the scheduled observations

        :return:
        """

        current_observation = self._head
        while current_observation is not None:
            yield current_observation
            current_observation = current_observation.next_observation

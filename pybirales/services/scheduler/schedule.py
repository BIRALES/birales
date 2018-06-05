import datetime
import logging as log

import pytz

from pybirales import settings
from pybirales.services.scheduler.exceptions import ObservationsConflictException, ObservationScheduledInPastException, \
    InvalidObservationException
from pybirales.services.scheduler.observation import ScheduledCalibrationObservation, ScheduledObservation, ScheduledAutoCalibrationObservation
from pybirales.utilities.source_transit import get_best_calibration_obs
from pybirales.events.publisher import EventsPublisher
from pybirales.events.events import ObservationScheduledEvent


class Schedule:
    def __init__(self):
        """
        The schedule representation of the scheduler

        :return:
        """
        self._head = None
        self._tail = None

        self.n_calibrations = 0
        self.n_observations = 0

        # Maximum amount of time allowed before the instrument needs to be re-calibrated (in hours)
        self._recalibration_time = datetime.timedelta(hours=settings.calibration.recalibration_time)

        # The time delta before source transits
        self._td_after_transit = datetime.timedelta(minutes=settings.calibration.transit_time_after)

        # The time delta after source transits
        self._td_before_transit = datetime.timedelta(minutes=settings.calibration.transit_time_before)

        self._publisher = EventsPublisher.Instance()

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
        else:
            raise InvalidObservationException('The Observation is neither a Calibration or a Space Debris Observation')

        # Fire an Observation was Scheduled Event
        self._publisher.publish(ObservationScheduledEvent(obs))

    def _conflicts(self, scheduled_observation, new_obs):
        """
        Check that the passed on observation does not conflict with the queued observations

        :param new_obs: The observation to be scheduled
        :type new_obs: ScheduledObservation
        :raises ObservationsConflictException: The observation conflicts with the queued observations
        :return: None
        """

        if scheduled_observation is None:
            log.debug('Observation `{}` does not overlap with the scheduled observations'.format(new_obs.name))
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

    def _is_valid(self, observation):
        """
        Determine if the observation is valid

        :param observation:
        :return:
        """
        try:
            # Check if observation is in the future
            observation.is_in_future()

            # Check if this observation conflicts with the scheduled observation
            self._conflicts(self._head, observation)
        except ObservationScheduledInPastException:
            return False
        except ObservationsConflictException:
            return False

        return True

    def add_observation(self, new_obs):
        """
        Add a new observation to the schedule

        :param new_obs:
        :return:
        """

        if not self._is_valid(new_obs):
            raise InvalidObservationException()

        if self._head is None:
            # Add this observation to the head of the schedule
            self._head = new_obs
            self._head.next_observation = self._tail

            # If a calibration observation is possible, schedule a calibration observation
            # before scheduling the (space debris) observation
            calibration_possible, calibration_obs = self._create_calibration_observation(new_obs)
            if calibration_possible:
                self._head = calibration_obs
                self._head.next_observation = new_obs
                self._increment(calibration_obs)

            self._tail = new_obs
            self._tail.next_observation = None
            self._tail.prev_observation = self._head

            self._increment(new_obs)

            return new_obs

        if new_obs.is_calibration_needed(self._head):
            # Get a new calibration observation and add it to the schedule
            calibration_possible, calibration_obs = self._create_calibration_observation(new_obs)

            if calibration_possible:
                self._append(calibration_obs)

        # Add the new observation whether a calibration observation is found or not
        self._append(new_obs)

        return new_obs

    def _append(self, new_obs):
        """
        Add the new observation to the tail of the schedule.

        :param new_obs: The observation to be added
        :return:
        """

        new_obs.prev_observation = self._tail
        new_obs.next_observation = None

        # The new observation's start time has to account for the time taken to move the antenna
        # new_obs.update_start_time_padding(self._tail)

        self._tail.next_observation = new_obs
        self._tail = new_obs

        self._increment(new_obs)

    def _create_calibration_observation(self, observation):
        """
        Return a calibration observation for the given observation.

        :param observation: The observation for which we want a calibration routine
        :return:
        """

        # Check if automatic calibration routines are disabled
        if not settings.scheduler.auto_calibrate:
            return False, None

        from_time = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        if observation.prev_observation:
            from_time = observation.prev_observation.end_time_padded

        # Get the available calibration sources on this day (with a time delta)
        available_sources = get_best_calibration_obs(from_time, observation.start_time_padded, self._td_after_transit,
                                                     self._td_before_transit)

        max_flux = -1
        calibration_obs = None

        for source in available_sources:
            # Create a calibration object for the closest_observation observation
            tmp_obs = ScheduledAutoCalibrationObservation(source['name'], source['transit_time'], observation.config_file)

            try:
                # Check if this observation conflicts with the scheduled observation
                self._conflicts(self._head, tmp_obs)

                if source['flux'] > max_flux:
                    calibration_obs = tmp_obs
            except ObservationsConflictException:
                log.debug("Calibration observation '{}' could not be scheduled".format(tmp_obs.name))

        if calibration_obs:
            log.debug('The `{}` observation was chosen for calibration.'.format(calibration_obs.name))
        else:
            log.warning('No suitable calibration source was found between {:%Y-%m-%dT%H:%M} and {:%Y-%m-%dT%H:%M}'
                        .format(from_time, observation.start_time_padded))

        return calibration_obs is not None, calibration_obs

    def __iter__(self):
        """
        Iterate over the scheduled observations

        :return:
        """

        current_observation = self._head
        while current_observation is not None:
            yield current_observation
            current_observation = current_observation.next_observation

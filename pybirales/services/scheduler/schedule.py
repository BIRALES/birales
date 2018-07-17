import datetime
import logging as log

import pytz

from pybirales.repository.models import Observation
from pybirales.services.scheduler.exceptions import ObservationsConflictException, InvalidObservationException
from pybirales.services.scheduler.observation import ScheduledCalibrationObservation, ScheduledObservation


class Schedule:
    def __init__(self):
        """
        The schedule representation of the scheduler

        :return:
        """
        self.observations = []

    @property
    def next_observation(self):
        """

        :return:
        """
        pending_observations = self.pending_observations()

        if pending_observations:
            return pending_observations[0]

        return None

    def _conflicts(self, new_obs):
        start_time = new_obs.start_time
        end_time = new_obs.end_time

        pending_observations = self.pending_observations()
        for observation in pending_observations:
            so_start = observation.start_time
            so_end = observation.end_time

            # Check for time ranges where duration is not defined:
            if not (so_end and end_time):
                # If there is an observation that has no end and starts before this observation
                if not so_end and so_start < start_time:
                    raise ObservationsConflictException(new_obs, observation)

                # If this observation starts before and does not end
                if not end_time and start_time < so_start:
                    raise ObservationsConflictException(new_obs, observation)

            # If this observation overlaps another time range
            elif (start_time <= so_end) and (so_start <= end_time):
                raise ObservationsConflictException(new_obs, observation)

        log.debug("Observation `%s` does not conflict with the scheduled observations", new_obs.name)

    def add(self, new_obs):
        """

        :param new_obs:
        :return:
        """

        try:
            # Check if observation is in the future
            new_obs.is_in_future()

            # Check if this observation conflicts with the scheduled observation
            self._conflicts(new_obs)
        except InvalidObservationException:
            log.warning('Observation %s is not valid. Could not add to the schedule', new_obs.name)
        else:
            # Save the observation to the database
            new_obs.save()

    def remove(self, obs):
        """

        :param obs:
        :return:
        """
        observation = Observation.objects.get(id=obs.id)
        observation.delete()

    def pending_observations(self):
        """

        :return:
        """
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        obs_models = Observation.get(from_time=now).order_by('-date_time_start')

        observations = []
        for o in obs_models:
            if o.type == 'calibration':
                observations.append(ScheduledCalibrationObservation(model_id=o.id))
            elif o.type == 'observation':
                observations.append(ScheduledObservation(model_id=o.id))
            else:
                raise InvalidObservationException('Observation type is not valid.')

        self.observations = observations

        return observations

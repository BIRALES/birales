class ObservationScheduledInPastException(Exception):
    def __init__(self, obs):
        Exception.__init__(self, "Cannot schedule the observation '{}' which starts in the past ({})"
                           .format(obs.name, obs.time_start_padded))
        self.observation = obs


class ObservationsConflictException(Exception):
    def __init__(self, obs, conflict_obs):
        Exception.__init__(self, "The observation '{}' overlaps the time range specified in observation '{}'".format(
            obs.name, conflict_obs.name))

        self.observation = obs
        self.conflict_obs = conflict_obs


class NoObservationsQueuedException(Exception):
    def __init__(self):
        Exception.__init__(self, "No observations are queued")


class IncorrectScheduleFormat(Exception):
    def __init__(self):
        Exception.__init__(self, "The format of the schedule is incorrect. "
                                 "Please ensure that the schedule is a valid JSON or TDM file")

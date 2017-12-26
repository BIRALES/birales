class ObservationScheduledInPastException(Exception):
    def __init__(self, obs):
        self.msg = "Cannot schedule the observation `{}` because it starts in the past ({})".format(
            obs.name, obs.start_time_padded)
        self.observation = obs
        Exception.__init__(self, self.msg)


class ObservationsConflictException(Exception):
    def __init__(self, obs, conflict_obs):
        self.msg = "The observation '{}' overlaps the time range specified in observation '{}'".format(
            obs.name, conflict_obs.name)
        self.observation = obs
        self.conflict_obs = conflict_obs

        Exception.__init__(self, self.msg)


class NoObservationsQueuedException(Exception):
    def __init__(self):
        self.msg = "No observations are queued"
        Exception.__init__(self, self.msg)


class IncorrectScheduleFormat(Exception):
    def __init__(self, schedule_file_path):
        self.msg = "Incorrect schedule format at {}. Please ensure that the schedule is a valid JSON or TDM file"\
            .format(schedule_file_path)
        Exception.__init__(self, self.msg)

import logging as log


class SchedulerException(Exception):
    pass

class NoObservationsQueuedException(SchedulerException):
    def __init__(self):
        self.msg = "No observations are queued"
        log.warning(self.msg)
        Exception.__init__(self, self.msg)


class IncorrectScheduleFormat(SchedulerException):
    def __init__(self, schedule_file_path):
        self.msg = "Incorrect schedule format at {}. Please ensure that the schedule is a valid JSON or TDM file" \
            .format(schedule_file_path)
        log.warning(self.msg)
        Exception.__init__(self, self.msg)


class InvalidObservationException(SchedulerException):
    def __init__(self, msg='Not specified'):
        self.msg = "Observation not valid. {}".format(msg)
        log.warning(self.msg)
        Exception.__init__(self, self.msg)

class ObservationScheduledInPastException(InvalidObservationException):
    def __init__(self, obs):
        self.msg = "Cannot schedule the observation `{}` because it starts in the past ({})".format(
            obs.name, obs.start_time)
        self.observation = obs

        log.warning(self.msg)
        Exception.__init__(self, self.msg)


class ObservationsConflictException(InvalidObservationException):
    def __init__(self, obs, conflict_obs):
        self.msg = "The observation '{}' overlaps the time range specified in observation '{}'".format(
            obs.name, conflict_obs.name)
        self.observation = obs
        self.conflict_obs = conflict_obs

        log.warning(self.msg)
        Exception.__init__(self, self.msg)


class IncorrectObservationParameters(InvalidObservationException):
    pass

class ObservationScheduledInPastException(Exception):
    def __init__(self, obs_name, obs_start):
        Exception.__init__(self, "Cannot schedule the observation '{}' which starts in the past ({})".format(obs_name,
                                                                                                             obs_start))


class ObservationsConflictException(Exception):
    def __init__(self, obs_name, overlapped_obs_name):
        Exception.__init__(self, "The observation '{}' overlaps the time range specified in observation '{}'".format(
            obs_name, overlapped_obs_name))


class NoObservationsQueuedException(Exception):
    def __init__(self):
        Exception.__init__(self, "No observations are queued")


class PipelineIsNotAvailableException(Exception):
    # A list of available pipelines
    AVAILABLE_PIPELINES = ['detection_pipeline', 'correlator_pipeline', 'standalone_pipeline']

    def __init__(self, pipeline_name):
        available_pipelines = "({})".format(', '.join(['%s'] * len(self.AVAILABLE_PIPELINES)))
        Exception.__init__(self, "The '{}' pipeline is not available. Please select one from the following list: {}"
                           .format(pipeline_name, available_pipelines))

import datetime
import logging as log
import os

from bson.objectid import ObjectId
from mongoengine import *

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
STATUS_MAP = {
    'pending': 'Scheduled',
    'running': 'Running',
    'finished': 'Finished',
    'error': 'Failed',
}


class BIRALESObservation(Document):
    meta = {'allow_inheritance': True, 'abstract': True}

    # _id = ObjectIdField(required=True, default=ObjectId, primary_key=True)
    name = StringField(required=True, max_length=200)
    date_time_start = DateTimeField(required=True)
    date_time_end = DateTimeField()  # Updated when the observation ends
    pipeline = StringField(required=True)
    type = StringField(default='observation')

    config_parameters = DynamicField(required=True)
    config_file = ListField(required=True)

    # These are updated when the observation has started
    settings = DynamicField()
    log_filepath = StringField()

    # These are updated during a detection observation
    noise_stats = DynamicField()
    sampling_time = FloatField()

    status = StringField()
    created_at = DateTimeField(required=True, default=datetime.datetime.utcnow)

    antenna_dec = FloatField()

    @queryset_manager
    def get(self, query_set, from_time=None, to_time=None, status=None):
        query = Q()

        if from_time:
            query &= Q(date_time_start__gte=from_time)

        if to_time:
            query &= Q(date_time_start__lte=to_time)

        if status:
            query &= Q(status__exact=status)

        return query_set.filter(query)

    def description(self):
        return {
            'status': self.status,
            'duration': self.config_parameters['duration'],
            'start': self.date_time_start,
            'end': self.date_time_end,
        }


class CalibrationObservation(BIRALESObservation):
    """
    A calibration observation
    """

    meta = {
        'collection': 'calibration'
    }

    type = StringField(default='calibration')
    real = ListField()
    imag = ListField()


class Observation(BIRALESObservation):
    """
    A detection Observation
    """
    meta = {
        'collection': 'observation'
    }

    tx = FloatField()
    calibration_obs = ReferenceField(CalibrationObservation)

    def description(self):
        return {
            'tx': self.tx,
            'status': self.status,
            'duration': self.config_parameters['duration'],
            'start': self.date_time_start,
            'end': self.date_time_end,
        }


class SpaceDebrisTrack(DynamicDocument):
    _id = ObjectIdField(required=True, default=ObjectId, primary_key=True)
    observation = ReferenceField(Observation, required=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow)

    m = FloatField(required=True)
    intercept = FloatField(required=True)
    r_value = FloatField(required=True)
    rcs = None
    target = None

    is_valid = BooleanField(default=True)

    tdm_filepath = StringField(required=False)

    @queryset_manager
    def get(self, query_set, observation_id=None, from_time=None, to_time=None):
        query = Q()

        if observation_id:
            query &= Q(observation=observation_id)

        if from_time:
            query &= Q(created_at__gte=from_time)

        if to_time:
            query &= Q(created_at__lte=to_time)

        return query_set.filter(query)

    def invalidate(self):
        self.is_valid = False


class BeamCandidate(DynamicDocument):
    _id = ObjectIdField(required=True, default=ObjectId, primary_key=True)
    observation = ReferenceField(Observation, required=True)
    beam_id = IntField(required=True)
    beam_ra = FloatField(required=True)
    beam_dec = FloatField(required=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow)

    min_time = DateTimeField(required=True)
    max_time = DateTimeField(required=True)
    max_channel = FloatField(required=True)
    min_channel = FloatField(required=True)

    @queryset_manager
    def get(self, query_set, observation_id=None, beam_id=None, max_channel=None, min_channel=None,
            to_time=None,
            from_time=None):

        query = Q()
        if observation_id:
            query &= Q(observation=observation_id)
        else:
            if beam_id:
                query &= Q(beam_id=beam_id)

            if from_time:
                query &= Q(min_time__gte=from_time)

            if to_time:
                query &= Q(max_time__lte=to_time)

            if min_channel:
                query &= Q(min_channel__gte=min_channel)

            if max_channel:
                query &= Q(max_channel__lte=max_channel)

        return query_set.filter(query)


class Event(DynamicDocument):
    _id = ObjectIdField(required=True, default=ObjectId, primary_key=True)
    name = StringField()
    channels = ListField()
    description = StringField()
    body = StringField()
    header = DynamicField()
    created_at = DateTimeField(required=True, default=datetime.datetime.utcnow)

    @queryset_manager
    def get(self, query_set, from_time=None, to_time=None):
        query = Q()

        if from_time:
            query &= Q(created_at__gte=from_time)

        if to_time:
            query &= Q(created_at__lte=to_time)

        return query_set.filter(query)


class Configuration(DynamicDocument):
    _id = ObjectIdField(required=True, default=ObjectId, primary_key=True)
    calibration_config_filepath = StringField()
    detection_config_filepath = StringField()

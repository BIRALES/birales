import datetime

from bson.objectid import ObjectId
from mongoengine import *

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
STATUS_MAP = {
    'pending': 'Scheduled',
    'running': 'Running',
    'finished': 'Finished',
}


class Observation(Document):
    name = StringField(required=True, max_length=200)
    date_time_start = DateTimeField(required=True, default=datetime.datetime.utcnow)
    date_time_end = DateTimeField()
    settings = DynamicField()
    noise_estimate = FloatField(default=0)

    @property
    def is_finished(self):
        now = datetime.datetime.utcnow()
        return now > self.date_time_end

    @property
    def is_running(self):
        return False

    @property
    def status(self):
        if self.is_finished:
            return STATUS_MAP['finished']

        if self.is_running:
            return STATUS_MAP['running']
        return STATUS_MAP['pending']

    def description(self):
        return {
            'tx': self.settings['observation']['transmitter_frequency'],
            'status': self.status,
            'duration': self.settings['observation']['duration'],
            'start': self.date_time_start,
            'end': self.date_time_end,
        }


class BeamCandidate(DynamicDocument):
    _id = ObjectIdField(required=True, default=ObjectId, unique=True, primary_key=True)
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


class SpaceDebrisTrack(DynamicDocument):
    _id = ObjectIdField(required=True, default=ObjectId, unique=True, primary_key=True)
    observation = ReferenceField(Observation, required=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow)

    m = FloatField(required=True)
    intercept = FloatField(required=True)
    score = FloatField(required=True)
    rcs = None
    target = None

    tdm_filepath = StringField(required=True)

    # data = DynamicDocument(required=True)

    @queryset_manager
    def get(self, query_set, observation_id=None):

        query = Q()
        if observation_id:
            query &= Q(observation=observation_id)

        return query_set.filter(query)

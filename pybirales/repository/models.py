from pybirales import settings
import datetime
import warnings
from mongoengine import *

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")


class Observation(Document):
    name = StringField(required=True, max_length=200)
    date_time_start = DateTimeField(required=True, default=datetime.datetime.utcnow)
    date_time_end = DateTimeField()
    settings = DynamicField()
    noise_estimate = FloatField(default=0)


class SpaceDebrisCandidate(Document):
    observation = ReferenceField(Observation, required=True)
    beam_id = IntField(required=True)
    beam_ra = FloatField(required=True)
    beam_dec = FloatField(required=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow)

    min_time = DateTimeField(required=True)
    max_time = DateTimeField(required=True)
    max_channel = FloatField(required=True)
    min_channel = FloatField(required=True)

    snr_data = ListField()
    channel_data = ListField()
    time_data = ListField()

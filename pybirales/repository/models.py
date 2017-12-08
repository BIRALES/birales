from pybirales import settings
import datetime
import warnings
from mongoengine import *

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


class Observation(Document):
    meta = {'allow_inheritance': True}
    name = StringField(required=True, max_length=200)
    date_time_start = DateTimeField(required=True, default=datetime.datetime.utcnow)
    date_time_end = DateTimeField()
    settings = DynamicField()

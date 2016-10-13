import logging as log
import pymongo as mongo
import sys

from helpers import DateTimeHelper
from configuration.application import config


class SpaceDebrisCandidate:
    """
    This class encapsulates the space debris detection as detected across multiple beams

    """

    # todo - merge beam space debris candidates into one
    def __init__(self):
        self.beam_space_debris = []

    def add(self, beam_space_debris):
        self.beam_space_debris.append(beam_space_debris)


class BeamSpaceDebrisCandidate:
    """
    This class encapsulates a space debris candidate detected in a beam
    """

    def __init__(self, name, beam, detection_data):
        self.beam = beam
        self.id = self.beam.observation_name + '.' + str(self.beam.id) + '.' + str(name)
        self.detection_data = detection_data

        self.data = {
            'time': [],
            'mdj2000': [],
            'time_elapsed': [],
            'frequency': [],
            'doppler_shift': [],
            'snr': [],
        }

        # todo - This function can be called lazily, since it is only for visualisation
        self.set_data(detection_data)

    def set_data(self, detection_data):
        for frequency, time, snr in sorted(detection_data, key=lambda row: row[1]):
            self.data['time'].append(time)
            self.data['mdj2000'].append(self._get_mdj2000(time))
            self.data['time_elapsed'].append(self._time_elapsed(time))
            self.data['frequency'].append(frequency)
            self.data['doppler_shift'].append(self._get_doppler_shift(self.beam.tx, frequency))
            self.data['snr'].append(snr)

    @staticmethod
    def _get_doppler_shift(transmission_frequency, reflected_frequency):
        return (reflected_frequency - transmission_frequency) * 1e6

    @staticmethod
    def _time_elapsed(time):
        return 'n/a'

    @staticmethod
    def _get_mdj2000(dates):
        return DateTimeHelper.ephem2juldate(dates)

    def save(self):
        if config.get('application', 'SAVE_CANDIDATES'):
            host = config.get('database', 'HOST')
            port = config.get('database', 'PORT')
            client = mongo.MongoClient(host, port)
            db = client['birales']

            data = {
                '_id': self.id,
                'data': self.data,
                'beam': self.beam.id,
                'observation': self.beam.observation_name,
                'data_set': self.beam.data_set.name,
                'tx': self.beam.tx,
            }

            try:
                key = {'_id': self.id}
                db.candidates.update(key, data, upsert=True)
            except mongo.errors.DuplicateKeyError:
                pass
            except mongo.errors.ServerSelectionTimeoutError:
                log.error('MongoDB is not running. Exiting.')
                sys.exit(1)

import logging as log
import pymongo as mongo
import sys
from abc import abstractmethod
from pybirales import settings
from pybirales.repository.models import BeamCandidate

class Repository:
    def __init__(self):
        self.host = 'localhost'
        self.port = 27017
        self.client = mongo.MongoClient(self.host, self.port)
        self.database = self.client['birales']

        self.client.birales.authenticate('birales_rw', 'arcadia10')

    @abstractmethod
    def persist(self, entity):
        pass


class BeamCandidateRepository(Repository):
    def __init__(self):
        Repository.__init__(self)
        self.collection = 'beam_candidates'

    def persist(self, beam_candidates):
        try:
            for candidate in beam_candidates:
                bc = BeamCandidate.from_json(candidate.to_json)
                bc.save()
        except Exception:
            log.exception('MongoDB is not running. Detected beam candidates could not be saved.')
        else:
            log.info('%s beam candidates were persisted', len(beam_candidates))

            for beam_candidate in beam_candidates:
                beam_candidate.to_save = False

    def get(self, beam_id=None, max_channel=None, min_channel=None, to_time=None, from_time=None):
        query = {}

        if beam_id:
            query['beam_id'] = beam_id

        if from_time:
            query['min_time'] = {'$gte': from_time}

        if to_time:
            query['max_time'] = {'$lte': to_time}

        if min_channel:
            query['min_channel'] = {'$gte': min_channel}

        if max_channel:
            query['max_channel'] = {'$lte': max_channel}

        try:
            beam_candidates = self.database['beam_candidates'].find(query).sort("min_time", mongo.ASCENDING)
        except Exception:
            log.exception('MongoDB is not running. Could not retrieve candidates.')
        else:
            return list(beam_candidates)

    def delete(self, beam_candidates):
        query = {
            '$or': [{'_id': beam_candidate.id} for beam_candidate in beam_candidates]
        }
        try:
            result = self.database.beam_candidates.remove(query)

        except Exception:
            log.exception('MongoDB is not running. Could not delete %s candidates.', len(beam_candidates))
        else:
            print(result)
            log.info('Deleted %s beam candidates', result['ok'])

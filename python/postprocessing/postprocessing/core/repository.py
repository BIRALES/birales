from abc import abstractmethod
from configuration.application import config
import pymongo as mongo
import logging as log
import sys


class Repository:
    def __init__(self, data_set):
        self.host = config.get('database', 'HOST')
        self.port = config.get_int('database', 'PORT')
        self.client = mongo.MongoClient(self.host, self.port)
        self.database = self.client['birales']
        self.data_set = data_set

    @abstractmethod
    def persist(self, entity):
        pass


class DataSetRepository(Repository):
    def __init__(self, observation):
        Repository.__init__(self, observation)

    def persist(self, data_set):
        try:
            self.database.data_sets.update({'_id': data_set.id},
                                           {"$set": dict(data_set)},
                                           upsert=True)
            log.info('Data set %s meta data was saved to database', self.data_set.name)
        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)


class BeamCandidateRepository(Repository):
    def __init__(self, observation):
        Repository.__init__(self, observation)
        self.collection = 'beam_candidates'

    def persist(self, beam_candidates):
        saved = 0
        data_set_id = self.data_set.id
        bulk = self.database.beam_candidates.initialize_ordered_bulk_op()

        try:
            saved = self.database.beam_candidates.insert_many(beam_candidates)
        # bulk.find({'_id': 1}).update({'$set': {'foo': 'bar'}})
        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)
        finally:
            return saved


class SpaceDebrisRepository(Repository):
    def __init__(self, observation):
        Repository.__init__(self, observation)

    def persist(self, space_debris_candidates):
        pass

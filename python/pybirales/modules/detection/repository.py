import logging as log
import pymongo as mongo
import sys

from abc import abstractmethod
from pybirales.base import settings


class Repository:
    def __init__(self):
        self.host = 'localhost'
        self.port = 27017
        self.client = mongo.MongoClient(self.host, self.port, connect=False)
        self.database = self.client['birales']
        self.client.birales.authenticate('birales_rw', 'arcadia10')

    @abstractmethod
    def persist(self, entity):
        pass


class ConfigurationRepository(Repository):
    def __init__(self):
        Repository.__init__(self)

    def persist(self, configuration):
        try:
            self.database.configurations.update_one({'_id': configuration['configuration_id']},
                                                    {"$set": configuration},
                                                    upsert=True)
            log.info('Configuration data was saved to the database')
        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)

    def get(self, data_set_id):
        try:
            return self.database.configurations.find_one({
                '_id': data_set_id
            })

        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)

    def destroy(self, data_set_id):
        """
        Delete this data set from the database

        @param data_set_id The id of the data_set that is to be deleted
        :return:
        """
        try:
            result = self.database.configurations.delete_many({"_id": data_set_id})
            log.info('Deleted %s data sets', result.deleted_count)

        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)


class BeamCandidateRepository(Repository):
    def __init__(self):
        Repository.__init__(self)
        self.collection = 'beam_candidates'

    def persist(self, beam_candidates):
        if not beam_candidates:
            log.warning('No beam space debris candidates were found.')
            return False

        try:
            # Get JSON representation of candidates
            to_save = [candidate.to_json() for candidate in beam_candidates]
            # Save candidates to the database
            # if len(beam_candidates) is 1:
            #     self.database.beam_candidates.insert_one(to_save[0])
            # else:
            #     self.database.beam_candidates.insert_many(to_save)
            for candidate in to_save:
                self.database.beam_candidates.insert_one(candidate)

        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Detected beam candidates could not be saved.')
        else:
            log.info('%s beam candidates were persisted', len(beam_candidates))

            for beam_candidate in beam_candidates:
                beam_candidate.to_save = False

    def get(self, beam_id=None, max_channel=None, min_channel=None, max_time=None, min_time=None):
        query = {}

        if beam_id:
            query['beam_id'] = beam_id

        if min_time:
            query['min_time'] = {'$gte': min_time}

        if max_time:
            query['max_time'] = {'$lte': max_time}

        if min_channel:
            query['min_channel'] = {'$gte': min_channel}

        if max_channel:
            query['max_channel'] = {'$lte': max_channel}

        try:
            beam_candidates = self.database['beam_candidates'].find(query).sort("min_time", mongo.ASCENDING)
        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Could not retrieve candidates.')
        else:
            return list(beam_candidates)

    def delete(self, beam_candidates):
        query = {
            '$or': [{'_id': beam_candidate.id} for beam_candidate in beam_candidates]
        }
        try:
            result = self.database.beam_candidates.delete_many(query)

        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Could not delete %s candidates.', len(beam_candidates))
        else:
            log.info('Deleted %s beam candidates', result.deleted_count)

import logging as log
import sys
from abc import abstractmethod
import pymongo as mongo
from configuration.application import config
from pyevtk.hl import pointsToVTK
import numpy as np


class Repository:
    def __init__(self):
        self.host = config.get('database', 'HOST')
        self.port = config.get_int('database', 'PORT')
        self.client = mongo.MongoClient(self.host, self.port)
        self.database = self.client['birales']

    @abstractmethod
    def persist(self, entity):
        pass


class BeamDataRepository(Repository):
    def __init__(self, beam_id=None, data_set=None):
        Repository.__init__(self)
        self.data_set = data_set
        self.beam_id = beam_id

    def persist(self, detections):
        try:
            # Clear the database of old data
            self.database.filtered_data.remove(
                {'beam_id': self.beam_id},
                {'data_set_id': self.data_set}
            )

            # Convert beam objects to a dict representation
            filtered_data = {
                'data_set_id': self.data_set.id,
                'beam_id': self.beam_id,
                'time': detections[:, 0].tolist(),
                'channel': detections[:, 1].tolist(),
                'snr': detections[:, 2].tolist()
            }

            # Insert beam data to the database
            self.database.filtered_data.insert(filtered_data)
            log.info('Filtered data saved to database')

        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)

    def get(self, beam_id, data_set_id):
        try:
            detections = self.database.filtered_data.find({"$and": [
                {'beam_id': beam_id},
                {'data_set_id': data_set_id}
            ]})

            return list(detections)

        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)

    def destroy(self, data_set_id):
        """
        Delete this beam data from the database

        @param data_set_id The id of the data_set that is to be deleted
        :return:
        """
        try:
            result = self.database.filtered_data.delete_many({"data_set_id": data_set_id})
            log.info('Deleted %s beam data detections', result.deleted_count)
        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)


class DataSetRepository(Repository):
    def __init__(self):
        Repository.__init__(self)

    def persist(self, data_set):
        try:
            self.database.data_sets.update_one({'_id': data_set.id},
                                               {"$set": dict(data_set)},
                                               upsert=True)
            log.info('Data set %s meta data was saved to the database', data_set.name)
        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)

    def get(self, data_set_id):
        try:
            return self.database.data_sets.find_one({
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
            result = self.database.data_sets.delete_many({"_id": data_set_id})
            log.info('Deleted %s data sets', result.deleted_count)

        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)


class BeamCandidateRepository(Repository):
    def __init__(self, data_set=None):
        Repository.__init__(self)
        self.collection = 'beam_candidates'
        self.data_set = data_set

    def persist(self, beam_candidates):
        if not beam_candidates:
            log.warning('No beam space debris candidates were found.')
            return False

        # self.save_to_vtk(beam_candidates)

        try:
            # Clear the database of old beam data candidates
            self.destroy(self.data_set.id)

            # Convert beam objects to a dict representation
            beam_candidates = [dict(candidate) for candidate in beam_candidates]

            # Insert candidates to the database
            saved = self.database.beam_candidates.insert_many(beam_candidates)
            log.info('%s beam candidates were saved to the database', len(saved.inserted_ids))
        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)

    @staticmethod
    def save_to_vtk(beam_candidates):
        for candidate in beam_candidates:
            time = []
            frequencies = []
            snr = []
            for detection in candidate.detections:
                time.append(detection['time_elapsed'])
                frequencies.append(detection['frequency'])
                snr.append(detection['snr'])

            pointsToVTK('beam_candidate_' + candidate.id,
                        x=np.array(time),
                        y=np.array(frequencies),
                        z=np.linspace(1., 1., len(frequencies)),
                        data={'snr': np.array(snr)})

    def get(self, beam_id, data_set_id):
        try:
            beam_candidates = self.database.beam_candidates.find({"$and": [
                {'beam_id': beam_id},
                {'data_set_id': data_set_id}
            ]})

            return list(beam_candidates)

        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)

    def destroy(self, data_set_id):
        try:
            result = self.database.beam_candidates.delete_many({"data_set_id": data_set_id})
            log.info('Deleted %s beam candidates', result.deleted_count)

        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)


class MultiBeamCandidateRepository(Repository):
    def __init__(self, data_set=None):
        Repository.__init__(self)
        self.collection = 'beam_candidates'
        self.data_set = data_set

    def get(self, data_set_id, max_freq=None, min_freq=None):
        try:
            multi_beam_candidates = []
            query = {'data_set_id': data_set_id}

            if max_freq and min_freq:
                query = {"$and": [
                    {'detections.frequency': {'$gte': float(min_freq)}},
                    {'detections.frequency': {'$lte': float(max_freq)}},
                    {'data_set_id': data_set_id}
                ]}

            beam_candidates = self.database.beam_candidates \
                .find(query) \
                .sort("illumination_time", mongo.ASCENDING)

            for candidate in list(beam_candidates):
                multi_beam_candidates.append(candidate)

            return multi_beam_candidates

        except mongo.errors.ServerSelectionTimeoutError:
            log.error('MongoDB is not running. Exiting.')
            sys.exit(1)

    def persist(self, entity):
        pass


class SpaceDebrisRepository(Repository):
    def __init__(self):
        Repository.__init__(self)

    def persist(self, space_debris_candidates):
        pass

import os.path
import config.application as config
import config.database as DB
import pymongo as mongo
import logging as log
import sys


class SpaceDebrisCandidateCollection:
    DB_FLAG = 'DB'

    def __init__(self, candidates):
        self.candidates = {}
        self.add(candidates)

    def __len__(self):
        return len(self.candidates)

    def add(self, candidates):
        for candidate in candidates:
            self.add_candidate(candidate)

    def add_candidate(self, candidate):
        self.candidates[id(candidate)] = candidate

    def drop_candidate(self, candidate):
        del self.candidates[id(candidate)]

    def view_candidates(self, output_dir, beam):
        detection_boxes = []
        detection_data = []
        for candidate_id, candidate in self.candidates.iteritems():
            detection_boxes.append(candidate.get_detection_box())
            detection_data.append(candidate.detection_data)

        beam_id = 'beam_' + str(beam.id)
        file_path = os.path.join(output_dir, beam_id, config.DETECTIONS_BEAM_FILE_NAME)

        beam.get_view().set_layout(figure_title = 'detections',
                                        x_axis_title = 'Frequency (Hz)',
                                        y_axis_title = 'Time (s)')

        beam.get_view().set_data(data = beam.snr.transpose(),
                                      x_axis = beam.channels,
                                      y_axis = beam.time)

        # beam.data.get_view().set_shapes(detection_boxes)
        beam.get_view().set_detections(detection_data)

        beam.get_view().save(file_path)

    def save_candidates(self, observation):
        if config.CANDIDATES_SAVE is self.DB_FLAG:
            self.save_candidates_db(observation)
        else:
            self.save_candidates_html(observation)

    def save_candidates_html(self, observation):
        for i, (candidate_id, candidate) in enumerate(self.candidates.iteritems()):
            beam_id = 'beam_' + str(candidate.beam.id)
            candidate_id = 'candidate_' + str(i)
            file_path = os.path.join(observation.beam_output_data, beam_id, 'candidates', candidate_id)

            if not os.path.exists(file_path):
                os.makedirs(file_path)

            # generate table
            candidate.create_table(file_path = os.path.join(file_path, config.OD_FILE_NAME), name = candidate_id)

    def save_candidates_db(self, observation):
        client = mongo.MongoClient(DB.HOST, DB.PORT)
        db = client['birales']
        for i, (c_id, candidate) in enumerate(self.candidates.iteritems()):
            candidate_id = observation.data_set + '.' + str(candidate.beam.id) + '.' + str(i)
            data = {
                '_id': candidate_id,
                'data': candidate.data,
                'beam': candidate.beam.id,
                'observation': observation.name,
                'data_set': observation.data_set,
                'tx': observation.tx,
            }
            try:
                key = {'_id': candidate_id}
                db.candidates.update(key, data, upsert= True)
            except mongo.errors.DuplicateKeyError:
                pass
            except mongo.errors.ServerSelectionTimeoutError:
                log.error('MongoDB is not running. Exiting.')
                sys.exit(1)

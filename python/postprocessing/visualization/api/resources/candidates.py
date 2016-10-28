from flask_restful import Resource
from flask import jsonify
from core.repository import BeamCandidateRepository, MultiBeamCandidateRepository


class MultiBeamCandidate(Resource):
    """
    The candidate that were detected in a beam
    """

    @staticmethod
    def get(observation, data_set):
        data_set_id = observation + '.' + data_set
        repository = MultiBeamCandidateRepository()
        data = repository.get(data_set_id)

        return jsonify(data)


class BeamCandidate(Resource):
    """
    The candidate that were detected in a beam
    """

    @staticmethod
    def get(observation, data_set, beam_id):
        data_set_id = observation + '.' + data_set
        repository = BeamCandidateRepository()
        data = repository.get(beam_id, data_set_id)

        return jsonify(data)


class SpaceDebrisCandidate(Resource):
    """
    A space debris candidate is a collection of beam candidates that is considered
    to be the space debris detections across multiple beams
    """

    def get(self):
        pass

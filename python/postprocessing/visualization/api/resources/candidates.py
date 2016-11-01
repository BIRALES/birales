from flask_restful import Resource
from flask import jsonify
from core.repository import BeamCandidateRepository, MultiBeamCandidateRepository


class MultiBeamCandidate(Resource):
    """
    The candidate that were detected in a beam
    """

    def get(self, observation, data_set):
        data_set_id = observation + '.' + data_set
        repository = MultiBeamCandidateRepository()
        candidates = repository.get(data_set_id)
        order = self._get_beam_illumination_order(candidates)

        data = {'candidates': repository.get(data_set_id),
                'order': order}

        return jsonify(data)

    @staticmethod
    def _get_beam_illumination_order(ordered_candidates):
        beam_order = [candidate['beam_id'] for candidate in ordered_candidates]
        return beam_order


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

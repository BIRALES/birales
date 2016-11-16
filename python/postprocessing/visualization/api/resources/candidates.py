from flask_restful import Resource
from flask import jsonify, request
from core.repository import BeamCandidateRepository, MultiBeamCandidateRepository, BeamDataRepository


class MultiBeamDetections(Resource):
    """
    The detections that were not filtered out when filters were applied to the beam
    """

    @staticmethod
    def get(observation, data_set, beam_id):
        data_set_id = observation + '.' + data_set
        repository = BeamDataRepository()
        beams_detections = repository.get(beam_id, data_set_id)
        filtered_data = []
        for detections in beams_detections:
            filtered_data.append({
                'data_set_id': detections['data_set_id'],
                'beam_id': detections['beam_id'],
                'time': detections['time'],
                'channel': detections['channel'],
                'snr': detections['snr']
            })

        return jsonify(filtered_data)


class MultiBeamCandidate(Resource):
    """
    The candidate that were detected in a beam
    """

    def get(self, observation, data_set):
        data_set_id = observation + '.' + data_set
        repository = MultiBeamCandidateRepository()

        max_freq = request.args.get('max_frequency')    # Max frequency in Mhz
        min_freq = request.args.get('min_frequency')    # Min frequency in Mhz

        candidates = repository.get(data_set_id, max_freq=max_freq, min_freq=min_freq)
        order = self._get_beam_illumination_order(candidates)

        data = {'candidates': candidates,
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

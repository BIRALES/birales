from flask_restful import Resource


class BeamCandidate(Resource):
    """
    The candidate that were detected in a beam
    """

    def get(self):
        pass


class SpaceDebrisCandidate(Resource):
    """
    A space debris candidate is a collection of beam candidates that is considered
    to be the space debris detections across multiple beams
    """

    def get(self):
        pass

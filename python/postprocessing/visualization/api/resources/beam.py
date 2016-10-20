from flask import jsonify
from flask_restful import Resource
from abc import abstractmethod


class Beam(Resource):
    """
    The beam parent class
    """

    @abstractmethod
    def get(self, observation, data_set, beam_id):
        pass


class RawBeam(Beam):
    """
    The beam data without any filtering applied
    """

    def get(self, observation, data_set, beam_id):
        # generate raw beam image
        # send raw image to client
        pass


class FilteredBeam(Beam):
    """
    The beam data after the filters were applied
    """

    def get(self, observation, data_set, beam_id):
        # generate filtered beam image
        # send filtered image to client
        pass

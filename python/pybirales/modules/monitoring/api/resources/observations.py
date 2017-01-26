from flask_restful import Resource
from flask import jsonify
from pybirales.modules.detection.repository import ObservationRepository


class Observations(Resource):

    @staticmethod
    def get():
        repository = ObservationRepository()
        data = repository.get()

        return jsonify(data)

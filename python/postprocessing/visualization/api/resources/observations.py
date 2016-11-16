from flask_restful import Resource
from flask import jsonify
from core.repository import ObservationRepository


class Observations(Resource):

    @staticmethod
    def get():
        repository = ObservationRepository()
        data = repository.get()

        return jsonify(data)

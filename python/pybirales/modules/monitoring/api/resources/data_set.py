from pybirales.modules.detection.repository import DataSetRepository
from flask import jsonify
from flask_restful import Resource
# from pybirales.configuration.application import config
from pybirales.base import settings


class DataSet(Resource):
    """
    The beam parent class
    """
    beam_dir = None
    image_ext = settings.monitoring.image_ext

    @staticmethod
    def get(observation, data_set):
        data_set_id = observation + '.' + data_set
        ds = DataSetRepository()
        data = ds.get(data_set_id)

        return jsonify(data)

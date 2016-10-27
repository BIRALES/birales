from flask_restful import Resource
from flask import abort, jsonify, Response
from configuration.application import config
from core.repository import DataSetRepository


class DataSet(Resource):
    """
    The beam parent class
    """
    beam_dir = None
    image_ext = config.get('visualization', 'IMAGE_EXT')

    @staticmethod
    def get(observation, data_set):
        data_set_id = observation + '.' + data_set
        ds = DataSetRepository()
        data = ds.get(data_set_id)

        return jsonify(data)

from flask_restful import Resource
from flask import send_file, abort
from abc import abstractmethod
from configuration.application import config

import os


class Beam(Resource):
    """
    The beam parent class
    """
    beam_dir = None
    image_ext = config.get('visualization', 'IMAGE_EXT')

    @abstractmethod
    def get(self, observation, data_set, beam_id, plot_type):
        pass

    @staticmethod
    def _send_beam_data(observation, data_set_name, file_name):
        file_path = os.path.join(config.ROOT,
                                 config.get('visualization', 'file_path'),
                                 observation,
                                 data_set_name,
                                 file_name)

        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/png')

        return abort(404)


class RawBeam(Beam):
    """
    The beam data without any filtering applied
    """

    def get(self, observation, data_set, beam_id, plot_type):
        if plot_type == 'water_fall':
            # build raw beam name
            file_name = 'raw_beam_' + str(beam_id) + self.image_ext
            # send raw image to client
            return self._send_beam_data(observation, data_set, file_name)


class FilteredBeam(Beam):
    """
    The beam data after the filters were applied
    """

    def get(self, observation, data_set, beam_id, plot_type):
        if plot_type == 'water_fall':
            # build filtered beam name
            file_name = 'filtered_beam_' + str(beam_id) + self.image_ext
            # send filtered image to client
            return self._send_beam_data(observation, data_set, file_name)

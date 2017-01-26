import os
import numpy as np

from abc import abstractmethod
from pybirales.modules.detection.data_set import DataSet
from pybirales.modules.detection.repository import BeamCandidateRepository
from flask import jsonify, request
from flask import send_file, abort
from flask_restful import Resource
# from pybirales.configuration.application import config


class Beam(Resource):
    """
    The beam parent class
    """
    beam_dir = None
    image_ext = config.get('monitoring', 'IMAGE_EXT')

    @abstractmethod
    def get(self, observation, data_set, beam_id, plot_type):
        pass

    @staticmethod
    def _send_beam_data(observation, data_set_name, file_name):
        file_path = os.path.join(config.ROOT,
                                 config.get('monitoring', 'file_path'),
                                 observation,
                                 data_set_name,
                                 file_name)

        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/png')

        return abort(404)


class RawBeam(Resource):
    """
    The beam data without any filtering applied
    """

    @staticmethod
    def _filter_beam_data(beam, max_freq, min_freq, max_time, min_time):
        channels_indices = \
            np.where(np.logical_and(min_freq <= beam.channels, beam.channels <= max_freq))[0]
        channels = beam.channels[channels_indices]

        time_indices = np.where(np.logical_and(min_time <= beam.time, beam.time <= max_time))[0]
        time = beam.time[time_indices]

        snr = beam.snr[time_indices, :][:, channels_indices]

        return {
            'channels': channels.tolist(),
            'time': time.tolist(),
            'snr': snr.tolist(),
        }

    def _get_beam_data(self, observation, data_set, beam_id, max_freq, min_freq, max_time, min_time):
        data_set = DataSet(observation, data_set, 32)
        beam = data_set.create_beam(beam_id)
        filtered_beam = data_set.create_beam(beam_id)
        filtered_beam.apply_filters()

        response = {
            'raw': self._filter_beam_data(beam, max_freq, min_freq, max_time, min_time),
            'filtered': self._filter_beam_data(filtered_beam, max_freq, min_freq, max_time, min_time)
        }

        return response

    @staticmethod
    def _get_candidates_in_beam(observation, data_set, beam_id, max_freq, min_freq, max_time, min_time):
        repository = BeamCandidateRepository()
        data_set_id = observation + '.' + data_set
        return repository.get(beam_id, data_set_id, max_freq=max_freq, min_freq=min_freq, max_time=max_time,
                              min_time=min_time)

    def get(self, observation, data_set, beam_id):
        max_freq = float(request.args.get('max_frequency'))  # Max frequency in Mhz
        min_freq = float(request.args.get('min_frequency'))  # Min frequency in Mhz
        max_time = float(request.args.get('max_time'))  # Max time in s
        min_time = float(request.args.get('min_time'))  # Min time in s

        beam_data = self._get_beam_data(observation, data_set, beam_id, max_freq, min_freq, max_time, min_time)

        response = {
            'raw_data': beam_data['raw'],
            'filtered_data': beam_data['filtered'],
            'candidates': self._get_candidates_in_beam(observation, data_set, beam_id, max_freq, min_freq, max_time,
                                                       min_time)
        }

        return jsonify(response)

    def get_image(self, observation, data_set, beam_id, plot_type):
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

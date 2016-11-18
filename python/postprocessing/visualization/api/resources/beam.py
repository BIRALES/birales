from flask_restful import Resource
from flask import send_file, abort
from abc import abstractmethod
from postprocessing.configuration.application import config
from postprocessing.core.data_set import DataSet
from postprocessing.core.repository import BeamCandidateRepository
from flask import jsonify, request

import os
import numpy as np


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


class RawBeam(Resource):
    """
    The beam data without any filtering applied
    """

    @staticmethod
    def _get_beam_data(observation, data_set, beam_id, max_freq, min_freq, max_time, min_time):
        data_set = DataSet(observation, data_set, 32)
        beam = data_set.beams[beam_id]

        channels_indices = \
            np.where(np.logical_and(min_freq <= beam.channels, beam.channels <= max_freq))[0]
        channels = beam.channels[channels_indices]

        time_indices = np.where(np.logical_and(min_time <= beam.time, beam.time <= max_time))[0]
        time = beam.time[time_indices]

        snr = beam.snr[time_indices, :][:, channels_indices]

        response = {
            'channels': channels.tolist(),
            'time': time.tolist(),
            'snr': snr.tolist(),
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

        response = {
            'raw_data': self._get_beam_data(observation, data_set, beam_id, max_freq, min_freq, max_time, min_time),
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

from flask import send_file
from flask import abort
from flask_restful import Resource
from abc import abstractmethod
from pybirales.modules.detection.data_set import DataSet
from pybirales.modules.monitoring.api.common.beam import MultiBeamVisualisation


class MultiBeam(Resource):
    """
    The multi-beam parent class. The multi-beam is a collection of beams generated by the backend
    """

    @staticmethod
    def _get_plot(bv, data, plot_type):

        if plot_type == 'bandpass':
            return send_file(bv.bandpass(data), mimetype='image/png')

        if plot_type == 'water_fall':
            return send_file(bv.waterfall(data), mimetype='image/png')

        # if plot type is not available, return 404 error
        abort(404)

    @abstractmethod
    def get(self, observation, data_set, plot_type):
        pass


class RawMultiBeam(MultiBeam):
    """
    The multi-beam without filtering
    """

    def get(self, observation, data_set, plot_type):
        bv = MultiBeamVisualisation(observation, data_set, name='raw_multi_beam')

        file_path = bv.get_plot_file_path(plot_type)
        if file_path:
            return send_file(file_path)

        data_set = DataSet(observation, data_set, n_beams=32)
        return self._get_plot(bv, data_set.beams, plot_type)


class FilteredMultiBeam(MultiBeam):
    """
    The multi-beam after filtering was applied
    """

    @staticmethod
    def _apply_filters(data_set):
        """
        Apply filters to the beam data
        :param data_set:
        :return:
        """
        for beam in data_set.beams:
            beam.apply_filters()
        return data_set

    def get(self, observation, data_set, plot_type):
        """
        Retrieve the Multi beam with filters applied
        :param observation:
        :param data_set:
        :param plot_type:
        :return:
        """
        data_set = DataSet(observation, data_set, n_beams=32)

        data_set = self._apply_filters(data_set)

        bv = MultiBeamVisualisation(beams=data_set.beams, name='filtered_multi_beam')

        return self._get_plot(bv, plot_type)

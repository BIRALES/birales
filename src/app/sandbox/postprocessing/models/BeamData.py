from abc import abstractmethod
from app.sandbox.postprocessing.views.BeamDataView import BeamDataView


class BeamData:
    def __init__(self):
        self.channels = None
        self.time = None
        self.snr = None
        self.name = None

        self._view = BeamDataView()

    def get_view(self):
        return self._view

    def view(self, file_path, name = 'Beam Data'):
        if name:
            self._view.name = name

        self._view.set_layout(figure_title = name,
                              x_axis_title = 'Frequency (Hz)',
                              y_axis_title = 'Time (s)')

        self._view.set_data(data = self.snr.transpose(),
                            x_axis = self.channels,
                            y_axis = self.time)

        self._view.save(file_path)

    @abstractmethod
    def set_data(self):
        pass

    def get_max_channel(self):
        return len(self.channels)

    def get_max_time(self):
        return len(self.time)

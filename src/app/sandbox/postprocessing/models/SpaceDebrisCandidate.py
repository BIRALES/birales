from app.sandbox.postprocessing.helpers.TableMakerHelper import TableMakerHelper
from app.sandbox.postprocessing.views.BeamDataView import BeamDataView
from app.sandbox.postprocessing.helpers.DateTimeHelper import DateTimeHelper


class SpaceDebrisCandidate:
    def __init__(self, tx, beam_id, detection_data):
        self.beam_id = beam_id
        self.tx = tx

        self.data = {
            'time': [],
            'mdj2000': [],
            'time_elapsed': [],
            'channel': [],
            'doppler_shift': [],
            'snr': [],
        }

        self.set_data(detection_data)

    def set_data(self, detection_data):
        for channel, time, snr in detection_data:
            self.data['time'].append(time)
            self.data['mdj2000'].append(SpaceDebrisCandidate.get_mdj2000(time))
            self.data['time_elapsed'].append(SpaceDebrisCandidate.time_elapsed(time))
            self.data['channel'].append(channel)
            self.data['doppler_shift'].append(SpaceDebrisCandidate.get_doppler_shift(self.tx, channel))
            self.data['snr'].append(snr)

    def view(self, beam_data, name = 'detection'):
        view = BeamDataView(name)
        view.set_layout(figure_title = name,
                        x_axis_title = 'Frequency (Hz)',
                        y_axis_title = 'Time (s)')

        for time, channel in zip(self.data['time'], self.data['channel']):
            snr = beam_data.snr[channel][time]
            # if snr == 1.:
            beam_data.snr[channel][time] = 2.0

        view.set_data(data = beam_data.snr.transpose(),
                      x_axis = beam_data.channels,
                      y_axis = beam_data.time
                      )
        view.show()

    def display_parameters(self, name = 'orbit_determination_input'):
        """
        Create table view
        :param name:
        :return:
        """
        table = TableMakerHelper()
        table.set_headers([
            'Epoch',
            'MJD2000',
            'Time Delay',
            'Frequency',
            'Doppler',
            'SNR'
        ])

        table.set_rows(self.data)
        table.build_table(name)

    @staticmethod
    def get_doppler_shift(tf, channel):
        return channel - tf

    @staticmethod
    def time_elapsed(time):
        return time

    @staticmethod
    def get_mdj2000(dates):
        return DateTimeHelper.ephem2juldate(dates)

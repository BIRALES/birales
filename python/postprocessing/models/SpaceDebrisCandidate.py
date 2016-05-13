import inflection as inf

from helpers.TableMakerHelper import TableMakerHelper
from helpers.DateTimeHelper import DateTimeHelper


class SpaceDebrisCandidate:
    # todo remove dependency on beam data
    def __init__(self, tx, beam, detection_data):
        self.beam = beam
        self.tx = tx
        self.detection_data = detection_data

        self.data = {
            'time': [],
            'mdj2000': [],
            'time_elapsed': [],
            'frequency': [],
            'doppler_shift': [],
            'snr': [],
        }

        self.set_data(detection_data)

    def set_data(self, detection_data):
        for channel, time, snr in detection_data:
            self.data['time'].append(time)
            self.data['mdj2000'].append(SpaceDebrisCandidate.get_mdj2000(time))
            self.data['time_elapsed'].append(SpaceDebrisCandidate.time_elapsed(time))
            self.data['frequency'].append(channel)
            self.data['doppler_shift'].append(SpaceDebrisCandidate.get_doppler_shift(self.tx, channel))
            self.data['snr'].append(snr)

    def view(self, file_path, name = 'Candidate'):
        view = self.beam.data.get_view()
        view.set_layout(figure_title = inf.humanize(name),
                        x_axis_title = 'Frequency (Hz)',
                        y_axis_title = 'Time (s)')

        # todo superimpose detection line as a scatter
        for time, channel in zip(self.data['time'], self.data['channel']):
            snr = self.beam.data.snr[channel][time]
            if snr == 1.:
                self.beam.data.snr[channel][time] = 2.0

        view.set_data(data = self.beam.data.snr.transpose(),
                      x_axis = self.beam.data.channels,
                      y_axis = self.beam.data.time
                      )
        view.show(file_path = file_path)

    def get_detection_box(self):
        x0 = self.data['frequency'][0]
        y0 = self.data['time'][0]

        x1 = self.data['frequency'][-1]
        y1 = self.data['time'][-1]

        shape = {
            'type': 'line',
            'x0': x0,
            'y0': y0,
            'x1': x1,
            'y1': y1,
            'line': {
                'color': 'rgba(255,255,255, 1)',
            },
        }

        return shape

    def create_table(self, file_path, name):
        """
        Create table view
        :param name:
        :param file_path:
        :return:
        """
        table = TableMakerHelper()

        table.set_caption(inf.humanize(name))

        table.set_headers([
            'Epoch',
            'MJD2000',
            'Time Delay',
            'Frequency',
            'Doppler',
            'SNR'
        ])

        table.set_rows(self.data)
        page = table.build_html_table(file_path)

        with open(file_path + '.html', 'w') as table:
            table.write(str(page))

    @staticmethod
    def get_doppler_shift(tf, channel):
        return tf - channel

    @staticmethod
    def time_elapsed(time):
        return time

    @staticmethod
    def get_mdj2000(dates):
        return DateTimeHelper.ephem2juldate(dates)

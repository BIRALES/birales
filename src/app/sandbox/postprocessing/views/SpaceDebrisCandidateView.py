from app.sandbox.postprocessing.helpers.TableMakerHelper import TableMakerHelper
from BeamDataView import BeamDataView


class SpaceDebrisCandidateView(BeamDataView):
    def __init__(self, name):
        BeamDataView.__init__(name)

        self.output_dir = 'public/results/'

    def save(self, rows, file_path = 'orbit_determination_input'):
        """
        Create table view
        :param rows:
        :param file_path:
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

        table.set_rows(rows)
        page = table.build_html_table(file_path)

        with open(self.output_dir + file_path + '.html', 'w') as table:
            table.write(str(page))

import logging as log
from pybirales import settings


class CalibrationFacade:

    def __init__(self):
        pass

    def calibrate(self):
        """
        Run the calibration Routine

        Adapted from TCPO / python / Scripts / Pipelines / CalibrationPipelineMultiProcessing.py

        :return:
        """

        # Settings from configuration file can be accessed like so:
        # json_file = settings.calibration.json_file
        print(settings.calibration.json_file)

        log.info('Running the calibration routine.')


    def _get_antenna_base_line(self):
        return True

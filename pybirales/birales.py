from pybirales.backend.services.calibrator.facade import CalibrationFacade


class BiralesFacade:

    def __init__(self):
        self._pipelines_subsystem = None
        self._calibration_subsystem = CalibrationFacade()

    def initialise(self):
        # Load configuration
        # Sanity Checks
        pass

    def run_pipeline(self):
        # Ensure status of the pipeline is correct
        # Check if calibration is required
        # Point the telescope
        # Start the chosen pipeline
        pass

    def calibrate(self):
        """
        Calibrate the instrument

        :return:
        """
        pass



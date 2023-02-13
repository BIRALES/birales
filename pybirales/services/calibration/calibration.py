import logging as log
import os
import pickle

import numpy as np

from pybirales.services.calibration import apply_coeffs
from pybirales.services.calibration import self_calibration
from pybirales.services.calibration import transit_select
from pybirales import settings


class CalibrationFacade:
    def __init__(self):
        # Create TM instance and load configuration json
        self._obs_time = None
        self.obs_info = None

    @staticmethod
    def load_pkl_file(filepath):
        """
        Load Pickle file

        :param filepath:
        :return:
        """

        try:
            return pickle.load(open(filepath + '.pkl', 'rb'))
        except IOError:
            raise Exception("PKL file not found in {}.pkl".format(filepath))

    def calibrate(self, calib_dir, corr_matrix_filepath):
        """
        Run the calibration Routine

        Adapted from TCPO / python / Scripts / Pipelines / CalibrationPipelineMultiProcessing.py

        :return:
        """

        # Load the observation settings only if the pipeline is running in offline mode
        self.obs_info = self.load_pkl_file(corr_matrix_filepath)

        log.info('Running the calibration routine.')
        calibration_config = {
            'no_of_antennas': len(settings.beamformer.antenna_locations),
            'gaincal': True,
            'phasecal': True,
            'stefcal': settings.calibration.stefcal,
            'pointing_ra': 0,
            'pointing_dec': settings.beamformer.reference_declination,
            'calibration_dec': settings.beamformer.reference_declination,
            'longitude': settings.beamformer.reference_antenna_location[0],
            'latitude': settings.beamformer.reference_antenna_location[1],
            'frequency': settings.observation.start_center_frequency * 1e6,
            'bandwidth': settings.observation.channel_bandwidth * 1e6,
            'main_dir': calib_dir,
            'obs_time': self.obs_info['timestamp'],
            'cal_coeffs_dir': calib_dir,
            'coeffs_filepath': os.path.join(os.environ['HOME'], '.birales/tcpo/calibration_coeffs',
                                            'coeffs__{}__{}.txt'.format(
                                                self.obs_info['settings']['observation']['name'],
                                                self.obs_info['timestamp'])),
            'model_generation': settings.calibration.model_generation,
            'test_run': settings.calibration.test_run_check,
            'transit_run': settings.calibration.transit_run,
            'obs_file': corr_matrix_filepath,
            'transit_file': corr_matrix_filepath,
            'calib_check_path': os.path.join(calib_dir, settings.calibration.calib_check_path),
            'model_vis_directory': calib_dir,
            'integration_time': settings.calibration.integration_time,
            'antennas': settings.beamformer.antenna_locations,
        }

        # Ensure that the directory structure exists.
        if not os.path.exists(os.path.dirname(calibration_config['calib_check_path'])):
            os.makedirs(os.path.dirname(calibration_config['calib_check_path']))

        # Generate baseline mapping
        cr = 0
        nof_baselines = np.int(0.5 * ((calibration_config['no_of_antennas'] ** 2) - calibration_config['no_of_antennas']))
        baseline_antenna_mapping = np.zeros((nof_baselines, 2), dtype=np.int)
        for i in range(calibration_config['no_of_antennas']):
            for j in range(calibration_config['no_of_antennas']):
                if i < j:
                    baseline_antenna_mapping[cr, 0] = i
                    baseline_antenna_mapping[cr, 1] = j
                    cr += 1

        calibration_config['baseline_no'] = baseline_antenna_mapping

        # Assign visibilities
        visibilies_file = calibration_config['transit_file']

        # Select transit source and time
        source_transit = transit_select.TransitSelect(calibration_config, visibilies_file)

        # Run self calibration
        calibration_run = self_calibration.SelfCalRun(calibration_config, source_transit.vis_in)

        # Assign instrumental coefficients (without geometric phases)
        coeffs_no_geom = np.array(calibration_run.selfcal.coeffs_no_geom)
        
        # Generate the visibilities (before and after) plot to check the calibration coefficients
        coefficient_manager = apply_coeffs.CoeffManagerRun(calibration_config, source_transit.vis_in,
                                                           calibration_run.selfcal.latest_coeffs)

        # Generate a fringe image for future checking
        fringe_image = coefficient_manager.check()

        # Return the instrumental coefficients and fringe image
        return coeffs_no_geom.real.flatten(), coeffs_no_geom.imag.flatten(), fringe_image


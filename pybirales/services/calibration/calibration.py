import logging as log
import os
import pickle

import numpy as np

import apply_coeffs
import self_calibration
import transit_select
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
            raise BaseException("PKL file not found in {}.pkl".format(filepath))

    def calibrate(self, calib_dir, corr_matrix_filepath):
        """
        Run the calibration Routine

        Adapted from TCPO / python / Scripts / Pipelines / CalibrationPipelineMultiProcessing.py

        :return:
        """

        # Load the observation settings only if the pipeline is running in offline mode
        self.obs_info = self.load_pkl_file(corr_matrix_filepath)

        log.info('Running the calibration routine.')

        cal_input = {
            'no_of_antennas': len(settings.beamformer.antenna_locations),
            'gaincal': True,
            'phasecal': True,
            'stefcal': settings.calibration.stefcal,
            'pointing_ra': 0,
            'pointing_dec': settings.beamformer.reference_declination,
            'calibration_dec': settings.beamformer.reference_declination,
            'longitude': settings.beamformer.reference_antenna_location[0],
            'latitude': settings.beamformer.reference_antenna_location[1],
            'frequency': (settings.observation.start_center_frequency + settings.observation.channel_bandwidth * 0.5) * 1e6,
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

        cr = 0
        no_of_baselines = np.int(0.5 * ((cal_input['no_of_antennas'] ** 2) - cal_input['no_of_antennas']))
        bas_ant_no = np.zeros((no_of_baselines, 2), dtype=np.int)
        for i in range(cal_input['no_of_antennas']):
            for j in range(cal_input['no_of_antennas']):
                if i < j:
                    bas_ant_no[cr, 0] = i
                    bas_ant_no[cr, 1] = j
                    cr += 1

        cal_input['baseline_no'] = bas_ant_no

        vis_file = cal_input['transit_file']

        transit_sel = transit_select.TransitSelect(cal_input, vis_file)
        calib_run = self_calibration.SelfCalRun(cal_input, transit_sel.vis_in)

        coeffs_no_geom = np.array(calib_run.selfcal.coeffs_no_geom)

        # Generate the visibilities (before and after) plot to check the calibration coefficients
        coeff_manager = apply_coeffs.CoeffManagerRun(cal_input, transit_sel.vis_in, calib_run.selfcal.coeffs_no_geom)

        fringe_image = coeff_manager.check()

        return coeffs_no_geom.real.flatten(), coeffs_no_geom.imag.flatten(), fringe_image

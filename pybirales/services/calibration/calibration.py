import logging as log
import os
import pickle
import time
import datetime
import numpy as np
from pytcpo.Core.DataModels.TelescopeModel import TelescopeModel as TM
from pytcpo.Core.PipeLine.PipelineParallelisation import PipelineParallelisation
from pytcpo.Pipelines.ModelVisibilitiesPipeline import ModelVisPipelineBuilder
from pytcpo.Pipelines.RealVisibilitiesPipeline import RealVisPipelineBuilder
from pybirales.pipeline.modules.persisters.corr_matrix_persister import create_corr_matrix_filepath

from pybirales import settings


class CalibrationFacade:
    def __init__(self, correlation_matrix_filepath=None):
        # Create TM instance and load configuration json
        self._tm = TM.Instance()
        self._obs_time = None
        self.obs_info = None
        self.dict_real = {}
        self.dict_imag = {}

        self.h5_filepath = correlation_matrix_filepath

    def _tcpo_config_adapter(self):
        antennas = {}
        ant_locations = np.array(settings.beamformer.antenna_locations)
        for i in range(ant_locations.shape[0]):
            antennas[np.str(i)] = [0, i, [ant_locations[i, 0], ant_locations[i, 1], ant_locations[i, 2]]]

        return {
            'antennas': antennas,
            'starting_channel': settings.observation.start_center_frequency,
            'bandwith': settings.observation.channel_bandwidth,
            'longitude_centre': settings.beamformer.reference_antenna_location[0],
            'latitude_centre': settings.beamformer.reference_antenna_location[1],
            'PCDec': settings.beamformer.reference_declination,
            't_average': settings.calibration.t_average,
            'tstep_length': settings.calibration.t_step_length,
            'obs_length': settings.calibration.obs_length,
            'start_time': self._get_obs_time().isoformat(),
        }

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

    def _get_real_vis_pipeline_parameters(self, config, corr_matrix_filepath, calib_dir, tm):
        """

        :param config:
        :param tm:
        :return:
        """

        time_steps = int(tm.ObsLength / tm.TimeAverage)
        no_of_antennas = tm.antennas.shape[0]
        bas_ant_no = self._get_antenna_base_line(config.auto_corr, no_of_antennas)

        return {'actual_vis_directory': calib_dir,
                'model_vis_directory': os.path.join(calib_dir, 'MSets'),
                'model_vis_list_path': os.path.join(calib_dir, 'model_vis.txt'),
                'total_time_samples': time_steps,
                'selfcal': config.selfcal,
                'stefcal': config.stefcal,
                'model_run': config.model_generation,
                'telescope': config.telescope,
                'phasecal': config.phasecal,
                'gaincal': config.gaincal,
                'auto_corr': config.auto_corr,
                'fescal': config.fescal,
                'no_of_antennas': no_of_antennas,
                'coeffs_filepath': os.path.join(calib_dir, 'coeffs_pointing.txt'),
                'transit_run': config.transit_run,
                'baseline_no': bas_ant_no,
                'coeff_test_run': config.coeff_test,
                'main_dir': calib_dir,
                'calibration_dec': config.calibration_dec,
                'pointing_ra': tm.PCRA,
                'pointing_dec': tm.PCDec,
                'antennas': tm.antennas,
                'longitude': tm.cen_lon,
                'latitude': tm.cen_lat,
                'calib_check_path': os.path.join(calib_dir, 'calib_plot.png'),
                'frequency': tm.StartFreq,
                'bandwith': tm.Bandwith,
                'calib_coeffs_dir':  os.path.join(os.environ['HOME'], config.calib_coeffs_dir),
                'obs_time': self._get_obs_time().isoformat(),
                'transit_file': corr_matrix_filepath,
                'obs_file': corr_matrix_filepath}

    @staticmethod
    def _get_antenna_base_line(auto_corr, no_of_antennas):
        bas_ant_no = None
        if auto_corr is True:
            cr = 0
            no_of_baselines = np.int(0.5 * ((no_of_antennas ** 2) + no_of_antennas))
            bas_ant_no = np.zeros((no_of_baselines, 2), dtype=np.int)
            for i in range(no_of_antennas):
                for j in range(no_of_antennas):
                    if i <= j:
                        bas_ant_no[cr, 0] = i
                        bas_ant_no[cr, 1] = j
                        cr += 1

        if auto_corr is False:
            cr = 0
            no_of_baselines = np.int(0.5 * ((no_of_antennas ** 2) - no_of_antennas))
            bas_ant_no = np.zeros((no_of_baselines, 2), dtype=np.int)
            for i in range(no_of_antennas):
                for j in range(no_of_antennas):
                    if i < j:
                        bas_ant_no[cr, 0] = i
                        bas_ant_no[cr, 1] = j
                        cr += 1

        return bas_ant_no

    @staticmethod
    def _get_calibration_coeffs(coeff_file):

        dict_real = {}
        dict_imag = {}

        calib_coeffs = np.loadtxt(coeff_file, dtype=np.complex)

        for i in range(len(calib_coeffs)):
            dict_real['a' + str(i)] = calib_coeffs[i].real
            dict_imag['a' + str(i)] = np.angle(calib_coeffs[i], deg=True)

        return dict_real, dict_imag

    @staticmethod
    def _get_reset_coeffs():
        dict_real = {}
        dict_imag = {}

        for i in range(len(settings.beamformer.antenna_locations)):
            dict_real['a' + str(i)] = 1.
            dict_imag['a' + str(i)] = 0.

        return dict_real, dict_imag

    def get_calibration_filepath(self):

        if self.h5_filepath:
            # If the correlated h5 file is provided, use that. (online or not)
            return os.path.dirname(self.h5_filepath), self.h5_filepath
        h5_filepath = create_corr_matrix_filepath(None)
        return os.path.dirname(h5_filepath), h5_filepath

    def _get_obs_time(self):
        """
        Get the observation time for this calibration run

        :return:
        """

        if not self._obs_time:
            if settings.manager.offline:
                # If we are running in offline mode (receiver disabled, reading from file)
                # create a new corr file with the OBSERVATION timestamp
                obs_info = self.load_pkl_file(self.h5_filepath)
                self._obs_time = obs_info['timestamp']
            else:
                # If we are NOT running in offline mode (receiver enabled, live data)
                # create a new corr file with the current timestamp
                self._obs_time = datetime.datetime.utcnow()
        return self._obs_time

    @property
    def real_reset_coeffs(self):
        return self._get_reset_coeffs()[0]

    @property
    def imag_reset_coeffs(self):
        return self._get_reset_coeffs()[1]

    def calibrate(self, calib_dir, corr_matrix_filepath):
        """
        Run the calibration Routine

        Adapted from TCPO / python / Scripts / Pipelines / CalibrationPipelineMultiProcessing.py

        :return:
        """

        # Configure the TM
        self._tm.from_dict(self._tcpo_config_adapter())

        # Load the observation settings only if the pipeline is running in offline mode
        self.obs_info = self.load_pkl_file(corr_matrix_filepath)

        log.info('Running the calibration routine.')

        model_vis_pipeline = ModelVisPipelineBuilder.ModelVisPipelineBuilder()
        model_vis_pipeline.setup(
            params={'TM_instance': self._tm, 'source': str(calib_dir),
                    'model_file': str(calib_dir + '/model_vis.txt'), 'local_sky_model': None})

        # RealVisGenPipeline process creation
        real_vis_pipeline = RealVisPipelineBuilder.RealVisPipelineBuilder()
        real_vis_pipeline.setup(
            params=self._get_real_vis_pipeline_parameters(settings.calibration, corr_matrix_filepath, calib_dir,
                                                          self._tm))

        real_vis_process = PipelineParallelisation(real_vis_pipeline.build())

        if settings.calibration.model_generation:
            model_vis_process = PipelineParallelisation(model_vis_pipeline.build())
            log.info('Starting Model Visibilities Process')
            model_vis_process.start()
            time.sleep(3)

        log.info('Starting Real Visibilities Process')
        real_vis_process.start()
        real_vis_process.join()

        log.info('Calibration routine finished')

        coeff_file = os.path.join(calib_dir, 'coeffs_no_geom.txt')

        self.dict_real, self.dict_imag = self._get_calibration_coeffs(coeff_file)

        coefficients = np.loadtxt(coeff_file, dtype=np.complex)

        # self._save_calibration_coeffs(coeff_file)

        return coefficients.real, coefficients.imag

    def _save_calibration_coeffs(self, coeff_file):
        try:
            tcpo_coefficients = np.loadtxt(coeff_file, dtype=np.complex)
        except IOError:
            log.error('TCPO Calibration coefficients were not found in %s', coeff_file)
        else:
            tcpo_filename = '{:%Y-%m-%dT%H:%M:%S}_{}.npy'.format(datetime.datetime.utcnow(),
                                                                 settings.beamformer.reference_declination)

            tcpo_filepath = os.path.join(os.environ['HOME'], settings.calibration.calib_coeffs_dir, tcpo_filename)

            np.save(tcpo_filepath, tcpo_coefficients)

            log.info('TCPO Calibration coefficients saved to %s', tcpo_filepath)

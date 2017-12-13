import logging as log
import os
import tempfile as tmp
import time
import shutil

from pytcpo.Pipelines.ModelVisibilitiesPipeline import ModelVisPipelineBuilder
from pytcpo.Pipelines.RealVisibilitiesPipeline import RealVisPipelineBuilder
from pytcpo.Core.PipeLine.PipelineParallelisation import PipelineParallelisation
from pytcpo.Core.DataModels.TelescopeModel import TelescopeModel as TM
import numpy as np
from pybirales import settings


class CalibrationFacade:
    def __init__(self):
        self._real_vis_dir = settings.calibration.real_vis_dir

    @staticmethod
    def _tcpo_config_adapter():
        antennas = {}
        ant_locations = np.array(settings.beamformer.antenna_locations)
        for i in range(ant_locations.shape[0]):
            antennas[np.str(i)] = [0, i, [ant_locations[i, 0], ant_locations[i, 1], ant_locations[i, 2]]]

        return {
            'antennas': antennas,
            'starting_channel': settings.observation.start_center_frequency,
            'bandwith': settings.observation.channel_bandwidth,
            'longitude_centre':  settings.beamformer.reference_antenna_location[0],
            'latitude_centre': settings.beamformer.reference_antenna_location[1],
            'PCDec': settings.beamformer.reference_declination,
            't_average': settings.calibration.t_average,
            'tstep_length': settings.calibration.t_step_length,
            'obs_length': settings.calibration.obs_length,
            'start_time': "21-11-2017 11:18:48.000",
        }

    def calibrate(self):
        """
        Run the calibration Routine

        Adapted from TCPO / python / Scripts / Pipelines / CalibrationPipelineMultiProcessing.py

        :return:
        """

        log.info('Running the calibration routine.')

        # Create temp dir
        main_dir = tmp.tempdir

        sc = settings.calibration

        # Create TM instance and load json
        tm = TM.Instance()
        tm.from_dict(self._tcpo_config_adapter())

        # Setup RealVisCal
        transit_run_file = os.path.join(self._real_vis_dir, sc.real_vis_file)
        calib_check_path = os.path.join(self._real_vis_dir, 'calib_plot.png')
        model_vis_dir = os.path.join(main_dir, 'MSets')
        model_vis_generated = os.path.join(main_dir, 'model_vis.txt')
        coeffs_out = os.path.join(main_dir, 'coeffs_pointing.txt')

        time_steps = int(tm.ObsLength / tm.TimeAverage)
        no_of_antennas = tm.antennas.shape[0]

        bas_ant_no = self._get_antenna_base_line(sc.auto_corr, no_of_antennas)

        # ModelVisGenPipeline process creation
        model_vis_pipeline = ModelVisPipelineBuilder.ModelVisPipelineBuilder()
        model_vis_pipeline.setup(
            params={'TM_instance': tm, 'source': main_dir, 'model_file': main_dir + '/model_vis.txt'})

        model_vis_manager = model_vis_pipeline.build()
        model_vis_process = PipelineParallelisation(model_vis_manager)

        # RealVisGenPipeline process creation
        real_vis_pipeline = RealVisPipelineBuilder.RealVisPipelineBuilder()
        real_vis_pipeline.setup(params={'actual_vis_directory': self._real_vis_dir + sc.real_vis_dir, 'model_vis_directory': model_vis_dir,
                                        'model_vis_list_path': model_vis_generated, 'total_time_samples': time_steps,
                                        'selfcal': sc.selfcal, 'stefcal': sc.stefcal, 'no_of_antennas': no_of_antennas,
                                        'coeffs_filepath': coeffs_out, 'transit_run': sc.transit_run,
                                        'baseline_no': bas_ant_no,
                                        'coeff_test_run': sc.coeff_test, 'main_dir': main_dir,
                                        'calibration_dec': sc.calibration_dec,
                                        'pointing_ra': tm.PCRA, 'pointing_dec': tm.PCDec, 'antennas': tm.antennas,
                                        'longitude': tm.cen_lon, 'latitude': tm.cen_lat,
                                        'calib_check_path': calib_check_path,
                                        'frequency': tm.StartFreq, 'bandwith': tm.Bandwith,
                                        'transit_file': transit_run_file})

        real_vis_manager = real_vis_pipeline.build()
        real_vis_process = PipelineParallelisation(real_vis_manager)

        if sc.model_generation is True:
            model_vis_process.start()
            time.sleep(3)
        real_vis_process.start()
        real_vis_process.join()

        shutil.rmtree(main_dir, ignore_errors=True)

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

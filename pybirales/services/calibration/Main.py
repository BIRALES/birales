from configparser import ConfigParser
import numpy as np
import distutils

import self_calibration
import apply_coeffs
import transit_select


class Main:

    def __init__(self):

        config_file = 'cal_config.ini'

        config = ConfigParser()
        config.read(config_file)

        cal_input = dict()

        cal_input['no_of_antennas'] = np.int(config.get('Configuration', 'no_of_antennas').encode('ascii', 'ignore'))
        cal_input['gaincal'] = bool(distutils.util.strtobool(config.get('Configuration', 'gaincal').encode('ascii', 'ignore')))
        cal_input['phasecal'] = bool(distutils.util.strtobool(config.get('Configuration', 'phasecal').encode('ascii', 'ignore')))
        cal_input['stefcal'] = bool(distutils.util.strtobool(config.get('Configuration', 'stefcal').encode('ascii', 'ignore')))
        cal_input['pointing_ra'] = np.float(config.get('Configuration', 'pointing_ra').encode('ascii', 'ignore'))
        cal_input['pointing_dec'] = np.float(config.get('Configuration', 'pointing_dec').encode('ascii', 'ignore'))
        cal_input['calibration_dec'] = np.float(config.get('Configuration', 'calibration_dec').encode('ascii', 'ignore'))
        cal_input['longitude'] = np.float(config.get('Configuration', 'longitude').encode('ascii', 'ignore'))
        cal_input['latitude'] = np.float(config.get('Configuration', 'latitude').encode('ascii', 'ignore'))
        cal_input['frequency'] = np.float(config.get('Configuration', 'frequency').encode('ascii', 'ignore'))
        cal_input['bandwidth'] = np.float(config.get('Configuration', 'bandwidth').encode('ascii', 'ignore'))
        cal_input['main_dir'] = config.get('Configuration', 'main_dir').encode('ascii', 'ignore')
        cal_input['obs_time'] = config.get('Configuration', 'obs_time').encode('ascii', 'ignore')
        cal_input['cal_coeffs_dir'] = config.get('Configuration', 'cal_dir').encode('ascii', 'ignore')
        cal_input['coeffs_filepath'] = config.get('Configuration', 'coeffs_filepath').encode('ascii', 'ignore')
        cal_input['model_generation'] = bool(distutils.util.strtobool(config.get('Configuration', 'model_generation').encode('ascii', 'ignore')))
        cal_input['test_run'] = bool(distutils.util.strtobool(config.get('Configuration', 'test_run_check').encode('ascii', 'ignore')))
        cal_input['transit_run'] = bool(distutils.util.strtobool(config.get('Configuration', 'transit_run').encode('ascii', 'ignore')))
        cal_input['obs_file'] = config.get('Configuration', 'obs_file').encode('ascii', 'ignore')
        cal_input['transit_file'] = config.get('Configuration', 'transit_file').encode('ascii', 'ignore')
        cal_input['calib_check_path'] = config.get('Configuration', 'calib_check_path').encode('ascii', 'ignore')
        cal_input['model_vis_directory'] = config.get('Configuration', 'model_vis_dir').encode('ascii', 'ignore')
        cal_input['integration_time'] = np.float(config.get('Configuration', 'integration_time').encode('ascii', 'ignore'))

        cal_input['antennas'] = {'0': [0, 0, [0.0, 0.0, 0.0]], '1': [0, 1, [5.6665, 0.0, 0.0]], '2': [0, 2, [11.333, 0.0, 0.0]],
                                 '3': [0, 3, [16.999, 0.0, 0.0]], '4': [0, 4, [0.0, 10.0, 0.0]],
                                 '5': [0, 5, [5.6665, 10.0, 0.0]], '6': [0, 6, [11.333, 10.0, 0.0]],
                                 '7': [0, 7, [16.999, 10.0, 0.0]], '8': [0, 8, [0.0, 20.0, 0.0]],
                                 '9': [0, 9, [5.6665, 20.0, 0.0]], '10': [0, 10, [11.333, 20.0, 0.0]],
                                 '11': [0, 11, [16.999, 20.0, 0.0]], '12': [0, 12, [0.0, 30.0, 0.0]],
                                 '13': [0, 13, [5.6665, 30.0, 0.0]], '14': [0, 14, [11.333, 30.0, 0.0]],
                                 '15': [0, 15, [16.999, 30.0, 0.0]], '16': [0, 16, [0.0, 40.0, 0.0]],
                                 '17': [0, 17, [5.6665, 40.0, 0.0]], '18': [0, 18, [11.333, 40.0, 0.0]],
                                 '19': [0, 19, [16.999, 40.0, 0.0]], '20': [0, 20, [0.0, 50.0, 0.0]],
                                 '21': [0, 21, [5.6665, 50.0, 0.0]], '22': [0, 22, [11.333, 50.0, 0.0]],
                                 '23': [0, 23, [16.999, 50.0, 0.0]], '24': [0, 24, [0.0, 60.0, 0.0]],
                                 '25': [0, 25, [5.6665, 60.0, 0.0]], '26': [0, 26, [11.333, 60.0, 0.0]],
                                 '27': [0, 27, [16.999, 60.0, 0.0]], '28': [0, 28, [0.0, 70.0, 0.0]],
                                 '29': [0, 29, [5.6665, 70.0, 0.0]], '30': [0, 30, [11.333, 70.0, 0.0]],
                                 '31': [0, 31, [16.999, 70.0, 0.0]]}

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
        print calib_run.selfcal.coeffs_no_geom
        apply_coeffs.CoeffManagerRun(cal_input, transit_sel.vis_in, calib_run.selfcal.coeffs_no_geom)


if __name__ == "__main__":
    Main()

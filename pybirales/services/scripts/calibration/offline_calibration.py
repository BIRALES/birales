import itertools
import os
import pickle

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pybirales.base.observation_manager import CalibrationObservationManager
from pybirales.pipeline.base.definitions import CalibrationFailedException
from pybirales.services.scheduler.observation import ScheduledCalibrationObservation


def calibrate(obs_root, config_filepath, parameters):
    source = find_source(obs_root)
    # obs_base = os.path.join(root, observation)

    try:
        corr_matrix_filepath = get_file_by_extension(obs_root, '_corr.h5')
        obs_info = pickle.load(open(get_file_by_extension(obs_root, '.pkl')))
    except IOError:
        print 'File not found in observation {}. Skipping observation'.format(obs_root)
        raise

    obs_name = source + '_' + obs_info['created_at'] + '_offline'

    parameters['beamformer']['reference_declination'] = obs_info['settings']['beamformer']['reference_declination']
    # parameters['beamformer']['reference_declination'] = 40

    print('Observation data path: {}'.format(obs_root))
    print('Correlation matrix: {}'.format(corr_matrix_filepath))
    print('Source: {}'.format(source))
    print('Observation name: {}'.format(obs_name))
    print('Declination is: {} deg'.format(parameters['beamformer']['reference_declination']))
    # continue
    # Create a results dir for calibration
    if not os.path.exists(os.path.join(obs_root, 'results')):
        os.makedirs(os.path.join(obs_root, 'results'))

    # Create a new calibration observation
    calibration_obs = ScheduledCalibrationObservation(name=obs_name,
                                                      pipeline_name='correlation_pipeline',
                                                      config_file=config_filepath,
                                                      config_parameters=parameters)
    # Initialise the calibration manager
    om = CalibrationObservationManager()

    om.run(observation=calibration_obs, corr_matrix_filepath=corr_matrix_filepath)

    model = calibration_obs.model

    return np.array(model.real), np.array(model.imag), source, obs_name, obs_info, corr_matrix_filepath


def find_source(obs_name):
    SOURCES = ['cas', 'cyg', 'tau', 'vir']
    for s in SOURCES:
        if s in obs_name.lower():
            return s

    raise BaseException("No viable source found in observation name")


def get_file_by_extension(observation_filepath, extension):
    for file in os.listdir(observation_filepath):
        if file.endswith(extension):
            return os.path.join(observation_filepath, file)

    raise IOError("Could not locate file with {} extension in {}".format(observation_filepath, extension))


def sources_map():
    return {
        'cas': 'Cassiopeia A',
        'cyg': 'Cygnus A',
        'tau': 'Taurus A',
        'vir': 'Virgo'
    }


def run_calibration_observations(observations, config_filepath, parameters):
    def get_antenna_id(n_antennas):
        cylinders = n_antennas // 4

        antenna_cyl = np.repeat(np.arange(1, cylinders + 1), 4)
        antenna_rec = np.tile([1, 2, 3, 4], cylinders)

        combined = np.array([antenna_cyl, antenna_rec]).T

        antenna_ids = []
        for c, r in combined:
            antenna_ids.append(str(c) + '-' + str(r))

        return antenna_ids

    df = pd.DataFrame(columns=['date', 'source', 'phase', 'amplitude'])

    labels = sources_map()
    for observation in observations:

        try:
            real, imag, source, obs_name, obs_info, corr_matrix_filepath = calibrate(
                os.path.join(DATA_ROOT, observation), config_filepath,
                parameters)
        except CalibrationFailedException:
            print 'Calibration {} failed. Skipping observation.'.format(observation)
            continue
        # except BaseException:
        #     print 'Something went wrong. Skipping observation.'
        #     continue
        else:
            complex_gain = real + imag * 1j

        amplitude = np.absolute(complex_gain)
        phase = np.angle(complex_gain, deg=True)

        obs_series = pd.Series(
            {'source': source,
             'date': obs_info['created_at'],
             'amplitude': amplitude,
             'amplitude_db': 10 * np.log10(amplitude),
             'antenna_id': get_antenna_id(32),
             'phase': phase},
            name=obs_name)

        df = df.append(obs_series)

        if PLOT_FRINGE_IMAGE:
            title = '{} on {:%d/%m/%Y at %H:%M}'.format(labels[source], obs_info['timestamp'])
            plot_fringe_image(corr_matrix_filepath, '',
                              title.replace(' ', '_').replace('/', '.') + '_validate_fringes.png')

    return df


def plot_fringe_image(cm_filepath, title, save_filepath):
    def read_corr_matrix(filepath):
        with h5py.File(filepath, "r") as f:
            return f["Vis"][:]

    def truncate_baselines_data(b_data):
        a = b_data[:, 1]
        limit = len(a) - np.argwhere(a[::-1] == 0)[-1][0] - 1

        return b_data[:limit, :]

    data_original = read_corr_matrix(cm_filepath)
    data_calib = read_corr_matrix(
        get_file_by_extension(os.path.join(os.path.dirname(cm_filepath), 'results'), 'calib.h5'))

    # plt.rcParams["figure.figsize"] = (16, 8)
    f, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(16, 10), constrained_layout=True)

    original_baselines = data_original[:, 0, :, 0].real
    corrected_baselines = data_calib[:, 0, :, 0].real

    original_baselines = truncate_baselines_data(original_baselines)
    corrected_baselines = truncate_baselines_data(corrected_baselines)

    for i in range(original_baselines.shape[1]):
        ax1.plot(original_baselines[:, i], linewidth=0.3)
        ax2.plot(corrected_baselines[:, i], linewidth=0.3)

    plt.suptitle(title)
    ax1.grid(alpha=0.3)
    ax2.grid(alpha=0.3)
    ax1.set_title('Uncalibrated')
    ax1.set_ylabel('Amplitude')

    ax2.set_title('Calibrated')
    ax2.set_xlabel('Time sample')
    ax2.set_ylabel('Amplitude')

    plt.savefig(save_filepath)

    return save_filepath


def visualise_calibration_coefficients(df, data_key, label, save_filepath):
    marker = {
        'cas': itertools.cycle(['b' + char for char in ['s', 'D', 'o', '^', 'v']]),
        'cyg': itertools.cycle(['g' + char for char in ['s', 'D', 'o', '^', 'v']]),
        'tau': itertools.cycle(['r' + char for char in ['s', 'D', 'o', '^', 'v']]),
        'vir': itertools.cycle(['k' + char for char in ['s', 'D', 'o', '^', 'v']]),
    }

    labels = sources_map()

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    # ax.xaxis.set_ticks(range(1, 33, 1))

    data_mean = np.zeros(32)
    for _, row in df.iterrows():
        data_mean += row[data_key]

    data_mean /= len(df)

    for _, row in df.iterrows():
        source = row['source']
        c, ms = next(marker[source])
        ax.plot(row['antenna_id'], row[data_key] - data_mean, marker=ms, color=c, markersize=12, ls='',
                markeredgewidth=1.0,
                markeredgecolor='black', label=labels[source])

    # ax.set_xlabel('Receiver', labelpad=20)
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    # ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=60)

    plt.savefig(save_filepath)


if __name__ == '__main__':

    OBSERVATIONS = []

    INVALID = [
        '2019_03_26/cyg_26_03_2019',
        '2019_03_27/cyg_27_03_2019',
        '2019_03_28/cyg',
        '2019_03_29/cyg',
        '2019_02_22/cas_22_02_2019',
        '2019_08_10/cas_raw',
        '2019_03_25/cas_25_03_2019',
        '2019_03_26/cas_26_03_2019',
        '2019_03_27/cas',
        '2019_03_28/cas',
        '2019_03_29/cas',
        '2019_02_05/cas_calib_05_02_2019',
    ]

    OLD = [
        '2019_02_05/tau_05_02_2019',
        # '2019_02_15/CASA_raw2',
        '2019_02_21/cas_21_02_2019',
        '2019_02_22/vir_21_02_2019',
        '2019_03_06/cyg_06_03_2019',
        # '2019_09_14/CASA',
    ]

    CONFIRMED = [
        # '2019_11_18/vir_a',  # 06:03
        # '2019_12_03/cas_03_12_2019',  # 16:51
        # '2019_12_03/cyg_03_12_2019',  # 13:45
        # '2019_12_03/tau_03_12_2019',  # 23:06
        # '2019_12_04/cas_04_12_2019',  # 16:49
        # '2019_12_04/tau_04_12_2019',  # 22:59
        # '2019_12_04/vir_04_12_2019',  # 06:03
        # '2019_12_05/cas_05_12_2019',  # 16:42
        # '2019_12_05/cyg_05_12_2019',  # 13:20
        # '2019_12_05/tau_05_12_2019',  # 22:55
        # '2019_12_05/vir_05_12_2019',  # 05:55
        # '2019_12_06/cas_06_12_2019',  # 16:38
        '2019_12_06/tau_06_12_2019',  # 22:51
        # '2019_12_06/vir_06_12_2019',  # 05:47
        # '2019_12_07/vir_07_12_2019',  # 05:43
    ]

    OBSERVATIONS += CONFIRMED

    # OBSERVATIONS += NEW

    PARAMETERS = {
        'manager': {
            'debug': True
        },
        'duration': 3600,
        'beamformer': {'reference_declination': 58.9}
    }

    CACHE = 'calibration_data_tau_only'
    PLOT_FRINGE_IMAGE = True
    CONFIG_ROOT = '/home/denis/.birales/configuration/'
    DATA_ROOT = '/media/denis/backup/birales/2019'
    config_filepath = [os.path.join(CONFIG_ROOT, 'birales.ini'),
                       os.path.join(CONFIG_ROOT, 'offline_calibration.ini')]

    if os.path.exists(CACHE):
        df = pd.read_pickle(CACHE)
    else:
        df = run_calibration_observations(observations=OBSERVATIONS, config_filepath=config_filepath,
                                          parameters=PARAMETERS)
        df.to_pickle(CACHE)

    visualise_calibration_coefficients(df, data_key='amplitude_db', label='Amplitude (dB)',
                                       save_filepath='calib_amp.png')
    visualise_calibration_coefficients(df, data_key='phase', label='Phase (deg)', save_filepath='calib_phase.png')

    plt.show()

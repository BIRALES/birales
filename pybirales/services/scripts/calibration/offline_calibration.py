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


def plot_fringe_image(cm_filepath, title, save_filepath):
    def read_corr_matrix(filepath):
        with h5py.File(filepath, "r") as f:
            return f["Vis"][:]

    data_original = read_corr_matrix(cm_filepath)
    data_calib = read_corr_matrix(
        get_file_by_extension(os.path.join(os.path.dirname(cm_filepath), 'results'), 'calib.h5'))

    # plt.rcParams["figure.figsize"] = (16, 8)
    f, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(16, 10))

    original_baselines = data_original[:, 0, :, 0].real
    corrected_baselines = data_calib[:, 0, :, 0].real

    for i in range(original_baselines.shape[1]):
        ax1.plot(original_baselines[:, i], linewidth=0.3)
        ax2.plot(corrected_baselines[:, i], linewidth=0.3)

    ax1.set_title(title + ' - Uncalibrated')
    ax1.set_ylabel('Amplitude')

    ax2.set_title(title + ' - Calibrated')
    ax2.set_xlabel('Time sample')
    ax2.set_ylabel('Amplitude')

    plt.savefig(save_filepath)

    return save_filepath


def run_calibration_observations(observations, parameters):
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
    for observation in observations:
        source = find_source(observation)
        obs_base = os.path.join(DATA_ROOT, observation)

        try:
            corr_matrix_filepath = get_file_by_extension(obs_base, '_corr.h5')
            obs_info = pickle.load(open(get_file_by_extension(obs_base, '.pkl')))
        except IOError:
            print 'File not found in observation {}. Skipping observation'.format(observation)
            continue

        obs_name = source + '_' + obs_info['created_at'] + '_offline'
        parameters['beamformer']['reference_declination'] = obs_info['settings']['beamformer']['reference_declination']
        # parameters['beamformer']['reference_declination'] = 40

        print('Observation data path: {}'.format(obs_base))
        print('Correlation matrix: {}'.format(corr_matrix_filepath))
        print('Source: {}'.format(source))
        print('Observation name: {}'.format(obs_name))
        print('Declination is: {} deg'.format(parameters['beamformer']['reference_declination']))
        # continue
        # Create a results dir for calibration
        if not os.path.exists(os.path.join(obs_base, 'results')):
            os.makedirs(os.path.join(obs_base, 'results'))

        # Create a new calibration observation
        calibration_obs = ScheduledCalibrationObservation(name=obs_name,
                                                          pipeline_name='correlation_pipeline',
                                                          config_file=config_filepath,
                                                          config_parameters=parameters)
        # Initialise the calibration manager
        om = CalibrationObservationManager()

        try:
            om.run(observation=calibration_obs, corr_matrix_filepath=corr_matrix_filepath)
        except CalibrationFailedException:
            print 'Calibration {} failed. Skipping observation.'.format(obs_name)
            continue

        model = calibration_obs.model

        complex_gain = np.array(model.real) + np.array(model.imag) * 1j
        print obs_name
        print complex_gain
        print
        amplitude = np.absolute(complex_gain)
        phase = np.angle(complex_gain, deg=True)

        obs_series = pd.Series(
            {'source': source,
             'date': obs_info['created_at'],
             'fringe_image_filepath': model.fringe_image,
             'amplitude': amplitude,
             'amplitude_db': np.log10(amplitude) * 10,
             'antenna_id': get_antenna_id(32),
             'phase': phase},
            name=obs_name)

        df = df.append(obs_series)

        plot_fringe_image(corr_matrix_filepath, obs_name, obs_name + '_validate_fringes.png')

    return df


def visualise_calibration_coefficients(df, data_key, label):
    marker = {
        'cas': itertools.cycle(['b' + char for char in ['s', 'D', 'o', '^', 'v']]),
        'cyg': itertools.cycle(['g' + char for char in ['s', 'D', 'o', '^', 'v']]),
        'tau': itertools.cycle(['r' + char for char in ['s', 'D', 'o', '^', 'v']]),
        'vir': itertools.cycle(['k' + char for char in ['s', 'D', 'o', '^', 'v']]),
    }

    labels = {
        'cas': 'Cassiopeia A',
        'cyg': 'Cygnus A',
        'tau': 'Taurus A',
        'vir': 'Virgo'
    }

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
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=60)


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

    CONFIRMED = [
        '2019_02_05/tau_05_02_2019',
        '2019_02_15/CASA_raw2',
        '2019_02_21/cas_21_02_2019',
        '2019_02_22/vir_21_02_2019',
        '2019_03_06/cyg_06_03_2019',
        '2019_09_14/CASA',
    ]

    OBSERVATIONS += CONFIRMED

    PARAMETERS = {
        'manager': {
            'debug': True
        },
        'duration': 3600,
        'beamformer': {'reference_declination': 58.9}
    }

    USE_CACHE = True
    CONFIG_ROOT = '/home/denis/.birales/configuration/'
    DATA_ROOT = '/media/denis/backup/birales/2019'
    config_filepath = [os.path.join(CONFIG_ROOT, 'birales.ini'),
                       os.path.join(CONFIG_ROOT, 'offline_calibration.ini')]
    RESULTS_FILENAME = 'calibration_data'

    if USE_CACHE:
        df = pd.read_pickle(RESULTS_FILENAME)
    else:
        df = run_calibration_observations(observations=OBSERVATIONS, parameters=PARAMETERS)
        df.to_pickle(RESULTS_FILENAME)

    visualise_calibration_coefficients(df, data_key='amplitude_db', label='Amplitude (dB)')
    visualise_calibration_coefficients(df, data_key='phase', label='Phase (deg)')

    plt.show()

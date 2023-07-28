import ctypes
import logging as log
import os
import pickle
import time
from abc import abstractmethod
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import njit, prange
from numpy import ctypeslib
from scipy import io

from pybirales.pipeline.modules.beamformer.pointing import Pointing
from pybirales.services.scripts.calibration.offline_calibration import calibrate

log.basicConfig(level=log.NOTSET)


@njit(parallel=True, fastmath=True)
def n_beamform(i, nof_beams, data, weights, output):
    for b in prange(nof_beams):
        x = np.dot(data, weights[0, b, :])
        output[b, i] = np.sum(np.power(np.abs(x), 2))

    return output


class Beamformer:
    def __init__(self):
        pass

    def beamform(self, config, calib_coeffs, pointing_weights, filepath):
        # Check filesize
        filesize = os.path.getsize(filepath)
        total_samples_per_antenna = filesize / (8 * nof_antennas)  # number of samples per antenna

        n_integrations = total_samples_per_antenna / nof_samples  # num. integrations per antenna

        n_chunks = int(n_integrations / (skip + 1))

        # Create output array
        output_data = np.zeros((config['nof_beams'], n_chunks), dtype=np.float64)

        output = self.output_blob(config, nof_samples, n_chunks)

        calib_coeffs = calib_coeffs[:nof_antennas_to_process]

        # Apply the weights
        weights = calib_coeffs * pointing_weights

        # Open file
        with open(filepath, 'rb') as f:
            processed_samples = 0
            t2 = time.time()
            for i in range(0, n_chunks):
                t1 = time.time()

                f.seek(nof_samples * nof_antennas * 8 * i * (skip + 1), 0)
                data = f.read(nof_samples * nof_antennas * 8)
                data = np.frombuffer(data, np.complex64)
                data = data.reshape((nof_samples, nof_antennas))
                data = np.ascontiguousarray(data[:, :nof_antennas_to_process], dtype=np.complex64)

                output_data = self._beamform(i, config['nof_beams'], data, weights, output, output_data)

                n_samples = (i + 1) * (skip + 1)

                # percentage = i / n_chunks * 100.
                progress = (i + 1) / float(n_integrations) * 100.
                log.info("Processed %d of %d samples [%.2f%%] in %.2f seconds. Skipped %d samples." % (
                    n_samples, n_integrations, progress, time.time() - t1, skip))

                processed_samples = nof_samples * nof_antennas * n_samples  # 32768*32
            else:
                log.info("Processed %d of %d samples in %.2f seconds" %
                         (processed_samples, filesize / 8., time.time() - t2))

        return output_data

    @abstractmethod
    def _beamform(self, i, nof_beams, data, weights, output, output_data):
        pass

    @abstractmethod
    def output_blob(self, config, nof_samples, nchunks):
        pass


class OfflineBeamformer(Beamformer):

    def __init__(self):
        Beamformer.__init__(self)

    # @profile
    def _beamform(self, i, nof_beams, data, weights, output, output_data):
        output_data = n_beamform(i, nof_beams, data, weights, output)
        return output_data

    def output_blob(self, config, nof_samples, nchunks):
        return np.zeros((config['nof_beams'], nchunks), dtype=np.complex64)


class PyBiralesBeamformer(Beamformer):

    def __init__(self):
        Beamformer.__init__(self)
        self.beamformer = ctypes.CDLL("/usr/local/lib/libbeamformer.so")
        complex_p = ctypeslib.ndpointer(np.complex64, ndim=1, flags='C')
        self.beamformer.beamform.argtypes = [complex_p, complex_p, complex_p, ctypes.c_uint32, ctypes.c_uint32,
                                             ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self.beamformer.beamform.restype = None

    def _beamform(self, i, nof_beams, input_data, weights, output, output_data):
        # input_data = data.reshape(1, 1, data.shape[0], data.shape[1])
        # output_data = output.reshape(1, 32, 1, 11)
        self.beamformer.beamform(input_data.ravel(), weights.ravel(), output.ravel(), input_data.shape[0], 1, nof_beams,
                                 nof_antennas_to_process,
                                 1, 8)

        output_data[:, i] = np.sum(np.power(np.abs(output), 2), axis=1)

        return output_data

    # def _beamform(self, input_data, weights, output, nof_samples, nof_subbands, nof_beams, nof_antennas_to_process):
    #     self.beamformer.beamform(input_data.ravel(), weights.ravel(), output.ravel(), nof_samples, 1, nof_beams,
    #                              nof_antennas_to_process,
    #                              1, 8)

    def output_blob(self, config, nof_samples, nchunks):
        return np.zeros((config['nof_beams'], nof_samples), dtype=np.complex64)


def get_weights(config, nof_antennas_to_process=32):
    # Create pointing object
    pointing = Pointing(config, 1, nof_antennas_to_process)

    # Generate pointings
    weights = pointing._pointing_weights

    return weights


def get_calibration_coefficients(mode='uncalibrated'):
    if mode == 'giu':
        return np.array([1.000000 + 0.000000j,
                         0.732380 - 0.588059j,
                         0.668125 - 0.269481j,
                         0.816918 - 0.567270j,
                         0.684808 - 0.699143j,
                         -0.659731 - 1.178600j,
                         1.143783 + 0.104276j,
                         -0.285521 - 1.068825j,
                         0.784679 + 0.219167j,
                         0.803180 - 0.455330j,
                         0.626265 + 0.787476j,
                         -0.252647 - 0.963801j,
                         0.823902 - 0.539061j,
                         0.914681 - 0.152378j,
                         0.314542 - 0.952656j,
                         0.555859 - 0.237510j,
                         0.495664 + 0.931462j,
                         1.043619 + 0.355521j,
                         0.696186 + 0.834885j,
                         0.975509 - 0.303480j,
                         0.638386 + 0.563067j,
                         0.086330 + 1.004608j,
                         0.991962 + 0.475933j,
                         0.877047 + 0.647834j,
                         0.855851 + 0.517210j,
                         0.510522 + 1.025221j,
                         0.952729 - 0.369845j,
                         0.966992 - 0.667751j,
                         0.235571 - 1.084553j,
                         0.779670 - 0.934907j,
                         0.947859 + 0.550121j,
                         0.157220 + 0.956486j], dtype=np.complex64)

    if mode == 'fes':
        # Calibration coefficients for CASA 14/09/2019
        return np.array([1.000000 + 0.000000j,
                         0.682469 - 0.824939j,
                         0.886676 - 0.603136j,
                         0.678056 - 0.891230j,
                         0.668060 - 0.833164j,
                         -0.791198 - 0.761666j,
                         1.127879 - 0.242153j,
                         -0.715554 - 0.785296j,
                         1.099111 - 0.003108j,
                         0.770427 - 0.838486j,
                         0.851895 + 0.494529j,
                         -0.755271 - 0.750812j,
                         0.799472 - 0.679416j,
                         0.948576 - 0.419620j,
                         -0.034330 - 1.063195j,
                         0.786582 - 0.968885j,
                         0.763981 + 0.895113j,
                         1.120969 + 0.033222j,
                         0.996422 + 0.608051j,
                         0.834703 - 0.806326j,
                         0.901297 + 0.611496j,
                         0.483737 + 1.131372j,
                         1.105564 - 0.010174j,
                         0.762407 + 1.028077j,
                         1.127753 + 0.274820j,
                         0.883949 + 0.711953j,
                         0.712730 - 0.928943j,
                         0.382533 - 1.144791j,
                         -0.094135 - 1.172973j,
                         0.319077 - 1.058574j,
                         1.095463 - 0.017532j,
                         0.949220 + 0.753913j], dtype=np.complex64)

    if mode == 'uncalibrated':
        return np.ones(shape=(32), dtype=np.complex64)

    if mode == 'stefcal':
        # this will be replaced by the calibration algorithm
        return np.ones(shape=(32), dtype=np.complex64)

    raise BaseException("Calibration mode is not valid")


def estimate_noise(b, output):
    return np.mean(np.concatenate([output[b, -50:], output[b, :100]]))


def visualise_bf_data(obs_name, skip, integration_time, output, beams, file_name=None, nof_antennas_to_process=32):
    fig, ax = plt.subplots(1)
    title = '{} beamformed data (Skip:{:d}, dt:{:0.2f} s, Antennas:{})'.format(obs_name, skip, integration_time,
                                                                               nof_antennas_to_process)
    samples = np.arange(0, output.shape[1], 1)
    time = integration_time * samples * (skip + 1)
    for b in beams:
        noise_estimate = estimate_noise(b, output)
        power = output[b, :] - noise_estimate
        power = power.real
        power[power <= 0] = np.nan
        power = 10 * np.log10(power)  # convert to db

        noise_db = 10 * np.log10(noise_estimate)
        ax.plot(time, power, label='Beam {:d}'.format(b))

        log.info("Antennas: {}, Beam: {}, Noise est. {:2.3f} dB, PSNR: {:2.3f}".format(nof_antennas_to_process, b, noise_db,
                                                                                       max(power) - noise_db))
    ax.set(xlabel='Time (s)', ylabel='Power (dB)', title=title)
    plt.legend()
    plt.grid()
    plt.show()

    if file_name:
        fig.savefig(file_name + '.png')


def display_obs_info(obs_name, obs_info, nof_samples, sampling_rate, integration_time):
    # start_time = obs_info['start_time']
    start_time = datetime.strptime(obs_info['start_time'][:-6], '%Y-%m-%d %H:%M:%S')
    duration = timedelta(seconds=obs_info['duration'])
    end_time = start_time + duration

    log.info("Observation `{}`".format(obs_name))
    log.info("Date: from {:%H:%M:%S} to {:%H:%M:%S %d/%m/%Y}".format(start_time, end_time))
    log.info("Duration: {:2.2f} minutes".format(duration.seconds / 60.))
    log.info("N samples per antenna: {:d}".format(nof_samples))
    log.info("Sampling rate: {:2.4f}".format(sampling_rate))
    log.info("Integration time: {:2.4f} seconds".format(integration_time))


def generate_csv(output, integration_time, skip, file_name):
    df = _generate_output(output, integration_time, skip)

    df.to_csv(file_name + '.csv', index=False, na_rep='NULL')


def generate_mat(output, integration_time, skip, nof_antennas, calibration_method, file_name):
    df = _generate_output(output, integration_time, skip)

    io.savemat(file_name, {
        'sample': [df[df['beam'] == b]['sample'].tolist() for b in beams],
        'time': [df[df['beam'] == b]['time'].tolist() for b in beams],
        'power': [df[df['beam'] == b]['power'].tolist() for b in beams],
        'beams': [b for b in beams],
        'power_db': [df[df['beam'] == b]['power_db'].tolist() for b in beams],
        'noise_est_db': [df[df['beam'] == b]['noise_est_db'].tolist() for b in beams],
        'skip': skip,
        'integration_time': integration_time,
        'nof_antennas': nof_antennas,
        'calibration_method': calibration_method,
        'units': 'seconds'
    }, do_compression=True)


def _generate_output(output, integration_time, skip):
    df = pd.DataFrame(columns=['sample', 'time', 'power', 'beam'])

    for b in range(0, output.shape[0]):
        samples = np.arange(0, output.shape[1], 1)
        noise_estimate = estimate_noise(b, output).real
        power = output[b, :] - noise_estimate
        power = power.real
        power[power <= 0] = np.nan
        power = 10 * np.log10(power)  # convert to db
        noise_db = 10 * np.log10(noise_estimate)

        df = df.append(pd.DataFrame({
            'sample': samples,
            'time': integration_time * samples * (skip + 1),
            'power': output[b][:].real,
            'beam': np.full(output.shape[1], b),
            'power_db': power,
            'noise_est_db': np.full(len(power), noise_db)
        }), ignore_index=True, sort=False)

    return df


def get_raw_filepaths(root_filepath):
    file_paths = []
    base_filepath = root_filepath.split('.')[0]

    next_filepath = root_filepath
    counter = 0
    while os.path.exists(next_filepath):
        log.info('Added: {}'.format(next_filepath))
        file_paths.append(next_filepath)

        next_filepath = '{}_{}.dat'.format(base_filepath, counter + 1)
        counter += 1

    return file_paths


def run(output_filepath):
    suffix = '{}_{}_{}'.format(calibration_mode, nof_antennas_to_process, skip)

    obs_root = os.path.abspath(os.path.join(obs_raw_file, os.pardir))
    obs_raw_name = os.path.basename(obs_raw_file)
    base_filepath = os.path.join(obs_root, obs_raw_name)
    # settings = pickle.load(open(base_filepath + '.pkl'))

    with open(base_filepath + '.pkl', 'rb') as pickle_file:
        settings = pickle.load(pickle_file)

    settings = settings['settings']
    obs_info = settings['observation']
    obs_name = obs_info['name']

    beamformer_config = settings['beamformer']
    beamformer_config['start_center_frequency'] = obs_info['start_center_frequency']
    beamformer_config['channel_bandwidth'] = obs_info['channel_bandwidth']

    sampling_rate = obs_info['samples_per_second']
    integration_time = nof_samples * 1. / sampling_rate  # integration time of each sample in seconds

    display_obs_info(obs_name, obs_info, nof_samples, sampling_rate, integration_time)

    pointing_weights = get_weights(beamformer_config, nof_antennas_to_process)

    if CALIBRATE:
        PARAMETERS = {
            'observation': {},
            'manager': {
                'debug': True
            },
            'duration': 3600,
            'beamformer': {'reference_declination': 58.9}
        }
        real, imag, source, obs_name, obs_info, _ = calibrate(obs_root, config_filepath, PARAMETERS)

        calib_coeffs = np.array(real + imag * 1j, dtype=np.complex64)
    else:
        calib_coeffs = get_calibration_coefficients(calibration_mode)

    filepaths = get_raw_filepaths(base_filepath)
    output_file_path = output_filepath + '{}_beamformed_data_{}'.format(obs_name, suffix)
    combined_output = []

    beamformer = OfflineBeamformer()
    # beamformer = PyBiralesBeamformer()

    if run_beamformer:
        t0 = time.time()
        time_elapsed = 0
        for j, filepath in enumerate(filepaths):
            # Run the beamformer on the input raw data
            output = beamformer.beamform(beamformer_config, calib_coeffs, pointing_weights, filepath)
            # output = beamformer.beamform(beamformer, beamformer_config, nof_samples, totalsamp, skip, calib_coeffs, pointing_weights,
            #                      nof_antennas_to_process)

            time_elapsed += time.time() - t0

            time_remaining = time_elapsed / (j + 1) * (len(filepaths) - (j + 1))
            log.info("Processed raw data at: {}. File {} of {}. Time elapsed: {:0.2f}. Remaining: {:0.2f}"
                     .format(filepath, j, len(filepaths), time_elapsed, time_remaining))

            combined_output.append(output)
        log.info("Beamforming finished in %.2f seconds" % (time.time() - t0))
        combined_output = np.hstack(combined_output)
    else:
        # Read the beamformed data
        combined_output = np.load(output_file_path + '.npy')

    if visualise:
        visualise_bf_data(obs_name, skip, integration_time, combined_output, beams, output_file_path, nof_antennas_to_process)

    if save_data:
        # Output data to csv file
        generate_csv(combined_output, integration_time, skip, output_file_path)

        generate_mat(combined_output, integration_time, skip, nof_antennas_to_process, suffix, output_file_path)

        if run_beamformer:
            # Output data as an numpy array
            np.save(output_file_path + '.npy', combined_output)


if __name__ == '__main__':
    # User defined parameters
    visualise = True
    CALIBRATE = True
    run_beamformer = True
    save_data = True

    nof_samples = 32768  # samples to integrate
    nof_antennas = 32  # number of antennas
    nof_antennas_to_process = 32
    calibration_mode = 'stefcal'
    skip = 0  # chunks to skip ( 0 does not skip)
    beams = [6, 15, 24, 30]  # beams to be plotted - central row
    beams = [7, 16, 25]
    beams = [5, 14, 23]

    CONFIG_ROOT = '/home/oper/.birales/configuration/'
    config_filepath = [os.path.join(CONFIG_ROOT, 'birales.ini'),
                       os.path.join(CONFIG_ROOT, 'offline_calibration.ini')]

    obs_raw_file = "/media/denis/backup/birales/2019/2019_09_14/CASA/CASA_raw.dat"
    # obs_raw_file = "/media/denis/backup/birales/2019/2019_08_14/CAS_A_FES/CAS_A_FES_raw.dat"
    obs_raw_file = '/storage/data/birales/2022_02_23/CasA/CasA_raw.dat'


    run('/storage/data/birales/2022_02_23/CasA/')

    # for n in [4, 8, 16, 32]:
    #     nof_antennas_to_process = n
    #     run()

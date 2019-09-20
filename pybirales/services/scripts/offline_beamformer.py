import logging as log
import os
import pickle
import time
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import njit, prange

from offline_pointing import Pointing

log.basicConfig(level=log.NOTSET)

# @profile
@njit(parallel=True, fastmath=True)
def beamformer(i, nbeams, data, weights, output):
    for b in prange(nbeams):
        x = np.dot(data, weights[0, b, :])
        output[b, i] = np.sum(np.power(np.abs(x), 2))


def get_weights(config, nants_to_process=32):
    # Create pointing object
    pointing = Pointing(config, 1, nants_to_process)

    # Generate pointings
    weights = pointing.weights

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

    raise BaseException("Calibration mode is not valid")


def beamform(config, nsamp, totalsamp, skip, calib_coeffs, pointing_weights, nants_to_process):
    total = totalsamp / nsamp

    n_chunks = (int(total) / skip) + 1

    # Create output array
    output = np.zeros((config['nbeams'], n_chunks), dtype=np.float)

    calib_coeffs = calib_coeffs[:nants_to_process]

    # Apply the weights
    weights = calib_coeffs * pointing_weights

    # Open file
    with open(filepath, 'rb') as f:
        for i in range(0, n_chunks):
            t1 = time.time()
            f.seek(nsamp * nants * 8 * i * skip, 0)
            data = f.read(nsamp * nants * 8)
            data = np.frombuffer(data, np.complex64)
            data = data.reshape((nsamp, nants))

            data = np.ascontiguousarray(data[:, :nants_to_process], dtype=np.complex64)

            # Perform calibration and beamforming
            beamformer(i, config['nbeams'], data, weights, output)

            # bbb = partial(bb, data, weights)
            # output[:, i] = np.array([c for c in pool.map(bbb, range(config['nbeams'])) if c])

            n_samples = (i + 1) * skip

            percentage = (i + 1.) / n_chunks * 100.
            log.info("Processing %d of %d [%.2f%%] dt=%.2f seconds" % (n_samples, total, percentage, time.time() - t1))


    return output


def visualise_bf_data(obs_name, skip, integration_time, output, beams, file_name=None, nants_to_process=32):
    fig, ax = plt.subplots(1)
    title = '{} beamformed data (Skip:{:d}, dt:{:0.2f} s, Antennas:{})'.format(obs_name, skip, integration_time,
                                                                               nants_to_process)
    samples = np.arange(0, output.shape[1], 1)
    time = integration_time * samples * skip
    for b in beams:
        ax.plot(time, 10 * np.log10(output[b, :]), label='Beam {:d}'.format(b))
    ax.set(xlabel='Time (s)', ylabel='Power (dB)', title=title)
    plt.legend()
    plt.grid()
    plt.show()

    if file_name:
        fig.savefig(file_name + '.png')


def display_obs_info(obs_info, nsamp, sampling_rate, integration_time):
    # start_time = obs_info['start_time']
    start_time = datetime.strptime(obs_info['start_time'][:-6], '%Y-%m-%d %H:%M:%S')
    duration = timedelta(seconds=obs_info['duration'])
    end_time = start_time + duration

    log.info("Observation `{}`".format(obs_name))
    log.info("Date: from {:%H:%M:%S} to {:%H:%M:%S %d/%m/%Y}".format(start_time, end_time))
    log.info("Duration: {:2.2f} minutes".format(duration.seconds / 60.))
    log.info("N samples per antenna: {:d}".format(nsamp))
    log.info("Sampling rate: {:d}".format(sampling_rate))
    log.info("Integration time: {:2.4f} seconds".format(integration_time))


def generate_csv(output, integration_time, skip, file_name):
    df = pd.DataFrame(columns=['sample', 'time', 'power', 'beam'])

    for b in range(0, output.shape[0]):
        samples = np.arange(0, output.shape[1], 1)
        df = df.append(pd.DataFrame({
            'sample': samples,
            'time': integration_time * samples * skip,
            'power': output[b][:],
            'beam': np.full(output.shape[1], b)
        }), ignore_index=True, sort=False)

    df.to_csv(file_name + '.csv')


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


if __name__ == '__main__':
    # User defined parameters
    visualise = True
    run_beamformer = True
    save_data = False

    nsamp = 32768  # samples to integrate
    nants = 32  # number of antennas
    nants_to_process = 32
    calibration_mode = 'fes'
    skip = 15  # chunks to skip
    beams = [6, 15, 24, 30]  # beams to be plotted
    # beams = [15]
    suffix = '{}_{}'.format(calibration_mode, nants_to_process)

    obs_raw_file = "/media/denis/backup/birales/2019/2019_09_14/CASA/CASA_raw.dat"
    # obs_raw_file = "/media/denis/backup/birales/2019/2019_08_14/CAS_A_FES/CAS_A_FES_raw.dat"

    obs_root = os.path.abspath(os.path.join(obs_raw_file, os.pardir))
    obs_raw_name = os.path.basename(obs_raw_file)
    base_filepath = os.path.join(obs_root, obs_raw_name)
    settings = pickle.load(open(base_filepath + '.pkl'))

    settings = settings['settings']
    obs_info = settings['observation']
    obs_name = obs_info['name']

    beamformer_config = settings['beamformer']
    beamformer_config['start_center_frequency'] = obs_info['start_center_frequency']
    beamformer_config['channel_bandwidth'] = obs_info['channel_bandwidth']

    sampling_rate = obs_info['samples_per_second']
    integration_time = nsamp * 1. / sampling_rate  # integration time of each sample in seconds

    display_obs_info(obs_info, nsamp, sampling_rate, integration_time)

    pointing_weights = get_weights(beamformer_config, nants_to_process)
    calib_coeffs = get_calibration_coefficients(calibration_mode)

    filepaths = get_raw_filepaths(base_filepath)
    output_file_path = 'beamformed_output/{}_beamformed_data_{}'.format(obs_name, suffix)
    combined_output = []

    if run_beamformer:
        t0 = time.time()
        for filepath in filepaths:
            log.info("Processing raw data at: {}".format(filepath))
            # Check filesize
            filesize = os.path.getsize(filepath)
            totalsamp = filesize / (8 * nants)  # number of samples per antenna

            # Run the beamformer on the input raw data
            output = beamform(beamformer_config, nsamp, totalsamp, skip, calib_coeffs, pointing_weights,
                              nants_to_process)

            combined_output.append(output)
        log.info("Beamforming finished in %.2f seconds" % (time.time() - t0))
    else:
        # Read the beamformed data
        output = np.load(output_file_path + '.npy')

    combined_output = np.hstack(combined_output)

    if visualise:
        visualise_bf_data(obs_name, skip, integration_time, combined_output, beams, output_file_path, nants_to_process)

    if save_data:
        # Output data to csv file
        generate_csv(combined_output, integration_time, skip, output_file_path)

        # Output data as an numpy array
        np.save(output_file_path + '.npy', output_file_path)

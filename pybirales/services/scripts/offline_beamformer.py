import os
import pickle
import time
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import njit, prange

from offline_pointing import Pointing


@njit(parallel=True, fastmath=True)
def beamformer(i, nbeams, data, weights, output):
    for b in prange(nbeams):
        x = np.dot(data, weights[0, b, :])
        output[b, i] = np.sum(np.power(np.abs(x), 2))


def get_weights(config):
    # Create pointing object
    pointing = Pointing(config, 1, 32)

    # Generate pointings
    weights = pointing.weights

    return weights


def get_calibration_coefficients():
    calibrated = np.array([1.000000 + 0.000000j,
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

    return np.ones(shape=calibrated.shape, dtype=np.complex64)


def beamform(config, nsamp, totalsamp, skip, calib_coeffs, pointing_weights):
    total = totalsamp / nsamp
    # Create output array
    output = np.zeros((config['nbeams'], int(total) / skip), dtype=np.float)

    # Apply the weights
    weights = calib_coeffs * pointing_weights

    # Open file
    with open(filepath, 'rb') as f:
        for i in range(0, int(total) / skip):
            t1 = time.time()

            f.seek(nsamp * nants * 8 * i * skip, 0)
            data = f.read(nsamp * nants * 8)
            data = np.frombuffer(data, np.complex64)
            data = data.reshape((nsamp, nants))

            # Perform calibration and beamforming
            beamformer(i, config['nbeams'], data, weights, output)
            n_samples = i * skip

            percentage = i / (float(total) / skip) * 100
            print("Processing %d of %d [%.2f%%] dt=%.2f seconds" % (n_samples, total, percentage, time.time() - t1))

    return output


def visualise_bf_data(obs_name, skip, integration_time, output, beams):
    fig, ax = plt.subplots(1)
    title = '{} beamformed data (Skip:{:d}, dt:{:0.2f} seconds)'.format(obs_name, skip, integration_time)
    samples = np.arange(0, output.shape[1], 1)
    time = integration_time * samples * skip
    for b in beams:
        ax.plot(time, 10 * np.log10(output[b, :]), label='Beam {:d}'.format(b))
    ax.set(xlabel='Time (s)', ylabel='Power (dB)', title=title)
    plt.legend()
    plt.grid()
    plt.show()


def display_obs_info(obs_info, nsamp, sampling_rate, integration_time):
    # start_time = obs_info['start_time']
    start_time = datetime.strptime(obs_info['start_time'][:-6], '%Y-%m-%d %H:%M:%S')
    duration = timedelta(seconds=obs_info['duration'])
    end_time = start_time + duration

    print "Observation `{}`".format(obs_name)
    print "Date: from {:%H:%M:%S} to {:%H:%M:%S %d/%m/%Y}".format(start_time, end_time)
    print "Duration: {:2.2f} minutes".format(duration.seconds / 60.)
    print "N samples per antenna: {:d}".format(nsamp)
    print "Sampling rate: {:d}".format(sampling_rate)
    print "Integration time: {:2.4f} seconds".format(integration_time)


def generate_csv(output, integration_time, skip, file_name):
    df = pd.DataFrame(columns=['sample', 'timestamp', 'power', 'beam'])

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
        file_paths.append(next_filepath)

        next_filepath = '{}_{}.dat'.format(base_filepath, counter + 1)
        counter += 1
        print next_filepath

    return file_paths


if __name__ == '__main__':
    # User defined parameters
    visualise = True
    run_beamformer = True
    nsamp = 32768  # samples to integrate
    nants = 32  # number of antennas
    skip = 15  # chunks to skip
    beams = [6, 15, 24, 30]  # beams to be plotted

    obs_raw_file = "/media/denis/backup/birales/2019/2019_09_14/CASA/CASA_raw.dat"
    obs_raw_file = "/media/denis/backup/birales/2019/2019_08_14/CAS_A_FES/CAS_A_FES_raw.dat"

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

    pointing_weights = get_weights(beamformer_config)
    calib_coeffs = get_calibration_coefficients()

    filepaths = get_raw_filepaths(base_filepath)
    output_file_path = 'beamformed_output/{}_beamformed_data'.format(obs_name)
    combined_output = []

    if run_beamformer:
        t0 = time.time()
        for filepath in filepaths:
            print "Processing raw data at: {}".format(filepath)
            # Check filesize
            filesize = os.path.getsize(filepath)
            totalsamp = filesize / (8 * nants)  # number of samples per antenna

            # Run the beamformer on the input raw data
            output = beamform(beamformer_config, nsamp, totalsamp, skip, calib_coeffs, pointing_weights)

            combined_output.append(output)
        print "Beamforming finished in %.2f seconds" % (time.time() - t0)
    else:
        # Read the beamformed data
        output = np.load(output_file_path + '.npy')

    combined_output = np.hstack(combined_output)

    if visualise:
        visualise_bf_data(obs_name, skip, integration_time, combined_output, beams)

    # Output data to csv file
    generate_csv(combined_output, integration_time, skip, output_file_path)

    # Output data as an numpy array
    np.save(output_file_path + '.npy', output_file_path)

import os
import pickle

import pandas as pd
from matplotlib import pyplot as plt
from mongoengine import connect
from sshtunnel import SSHTunnelForwarder

from pybirales.pipeline.modules.detection.msds.util import timeit
from pybirales.repository.models import Observation, SpaceDebrisTrack


def get_observation(type, duration):
    if type == 'observation':
        observations = Observation.objects(class_check=False, type=type, __raw__={"settings.manager.offline": False},
                                           config_parameters__duration__gte=duration).order_by(
            '-date_time_start')
    else:
        observations = Observation.objects(class_check=False, type=type,
                                           config_parameters__duration__gte=duration).order_by('-date_time_start')
    return observations


@timeit
def get_tracks(cache, use_cache=False):
    if use_cache:
        return pickle.load(open(cache, "rb"))

    tracks = SpaceDebrisTrack.objects(track_size__gte=50, r_value__lte=-0.99, duration__gte=10)

    df = pd.DataFrame(columns=['created_at', 'm', 'r_value'])
    for t in tracks:
        df = df.append({
            'created_at': t.created_at, 'm': t.m, 'r_value': t.r_value
        }, ignore_index=True)

    pickle.dump(df, open(cache, "wb"))

    return tracks


def get_total_obs_time(observations):
    total_time = 0
    for obs in observations:
        total_time += obs.config_parameters['duration']
    return total_time / 3600.


def is_calib(obs_name):
    for s in CALIB_TYPES:
        if s in obs_name.lower():
            return True
    return False


def get_raw_filepaths(root):
    raw_files = {
        'detection': [],
        'calibration': []
    }
    for dir_name, _, fileList in os.walk(root):
        for f in fileList:
            if f.endswith('_raw.dat'):
                if is_calib(f):
                    raw_files['calibration'].append(os.path.join(dir_name, f))
                else:
                    raw_files['detection'].append(os.path.join(dir_name, f))

    return raw_files


def obs_time_from_raw(obs_filepath):
    # raw_file = open(obs_filepath, 'rb')
    # n_blobs = os.stat(obs_filepath).st_size / (nsamp * nants * 8)
    # dt = nsamp / sampling_rate
    # nsamp = 262144
    nants = 32
    sampling_rate = 78125

    return os.stat(obs_filepath).st_size / (sampling_rate * nants * 8)


def get_total_time(raw_files):
    total_time = 0
    for f in raw_files:
        total_time += obs_time_from_raw(f)

    return total_time / 3600.


def raw_files_msg(raw_files, obs_type):
    data = raw_files[obs_type]
    n_files = len(data)
    tot_time = get_total_time(data)
    return "Found {} {} raw files, totalling {:0.3f} hours of observation time.".format(n_files, obs_type, tot_time)


def obs_msg(observations, obs_type):
    n_obs = len(observations)
    tot_time = get_total_obs_time(observations)
    return "Found {} {} observations, totalling {:0.3f} hours of observation time.".format(n_obs, obs_type, tot_time)


@timeit
def plot_tracks(df):
    print("Found {} tracks. From {:%d/%m/%y} to {:%d/%m/%y}".format(len(df), df['created_at'].min(),
                                                                    df['created_at'].max()))

    df = df.set_index('created_at')
    df = df.resample('M').count()

    fig, ax = plt.subplots(1, figsize=(11, 8))
    x_labels = ['{:%m/%y}'.format(pd.to_datetime(str(d))) for d in df.index.values]
    ax.bar(x_labels, df.m, color='#e0e0e0', zorder=3, edgecolor='#666666', alpha=0.8)

    ax.grid(alpha=0.3)
    ax.set_ylabel('Number of detected tracks')
    # ax.set_xlabel('Date')
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.tick_params(axis='x', labelsize=10)
    plt.tight_layout()

    plt.savefig('tracks_histogram_{}.pdf'.format(len(df)))

    return ax


if __name__ == '__main__':
    USERNAME = os.environ.get('FAHAL_USERNAME')
    PASSWORD = os.environ.get('FAHAL_PASSWORD')
    DATABASE = os.environ.get('FAHAL_DB')
    PORT = int(os.environ.get('FAHAL_PORT'))
    HOST = os.environ.get('FAHAL_HOST')

    SSH_USER = os.environ.get('FAHAL_SSH_USER')
    SSH_PASS = os.environ.get('FAHAL_SSH_PASS')

    CALIB_TYPES = ['cas', 'cyg', 'tau', 'vir']
    CACHE = 'cache.pkl'

    server = SSHTunnelForwarder(
        (HOST, 22),
        ssh_username=SSH_USER,
        ssh_password=SSH_PASS,
        remote_bind_address=('localhost', 27017)
    )

    server.start()

    db_connection = connect(
        db=DATABASE,
        username=USERNAME,
        password=PASSWORD,
        host='127.0.0.1',
        port=server.local_bind_port
    )

    try:

        # raw_files = get_raw_filepaths(root='/media/denis/backup/birales/') # to be run on fahal for prod values
        #
        # print raw_files_msg(raw_files, 'detection')
        # print raw_files_msg(raw_files, 'calibration')
        #
        detect_obs = get_observation(type='observation', duration=600)
        calibration_obs = get_observation(type='calibration', duration=1000)

        print(obs_msg(detect_obs, 'detection'))
        print(obs_msg(calibration_obs, 'calibration'))

        tracks_df = get_tracks(CACHE, use_cache=True)
        ax = plot_tracks(tracks_df)

        plt.show()

    except KeyboardInterrupt:
        print("Exiting.")
    finally:
        print("Tear down")
        db_connection.close()
        server.stop_module()

import datetime
import os

import pandas as pd

FILES = ['data.merged.tree', 'data2017.tree']
EXT = '_raw.dat'
LOCAL_DATA_DIR = '/media/denis/backup/birales/'
REMOTE_DATA_DIRS = ['/data/birales/', '/data2/birales/', '/data2/']


def get_target_observations(o):
    t_obs = []
    for obs in o:
        if 'norad' not in obs.lower():
            if 'tiangong' not in obs.lower():
                continue
        t_obs.append(obs)

    obs_df = pd.DataFrame(columns=['date', 'obs_name', 'raw_filepath'])
    # sort the observations by date
    for t in t_obs:
        splits = t.split('/')
        date_folder = splits[-3]
        date = datetime.datetime.strptime(date_folder, "%Y_%m_%d")

        obs_df = obs_df.append({
            'date': date,
            'obs_name': splits[-2].lower(),
            'raw_filepath': t
        }, ignore_index=True)

    return obs_df


def get_obs_type(file_str, o):
    for s in o.keys():
        if s in file_str.lower():
            return s
    return 'rso'


def get_synced_files(rso_files):
    synced_files = {
        'local_and_remote': [],
        'remote_only': [],
        'local_only': [],
        'all_local': [],
        'all_remote': []
    }

    for raw_rso_file in rso_files:
        for remote_root in REMOTE_DATA_DIRS:
            if raw_rso_file.startswith(remote_root):
                truncated_raw_rso_file = raw_rso_file[len(remote_root):]

                if os.path.exists(os.path.join(LOCAL_DATA_DIR, truncated_raw_rso_file)):
                    synced_files['local_and_remote'].append(raw_rso_file)
                    synced_files['all_local'].append(os.path.join(LOCAL_DATA_DIR, truncated_raw_rso_file))
                    synced_files['all_remote'].append(raw_rso_file)
                else:
                    synced_files['remote_only'].append(raw_rso_file)
                    synced_files['all_remote'].append(raw_rso_file)
                break
        else:
            synced_files['local_only'].append(raw_rso_file)
            synced_files['all_local'].append(raw_rso_file)

    return synced_files


def get_observations():
    o = {'cas': [], 'cyg': [], 'tau': [], 'vir': [], 'rso': [], 'test': []}
    for _file in FILES:
        with open(_file, 'rb') as f:
            for line in f:
                line = line.rstrip()  # strip out all tailing whitespace
                if not line.endswith(EXT):
                    continue

                obs_type = get_obs_type(line, o)
                o[obs_type].append(line)

    return o


def output_birales_stats(o):
    for k in o.keys():
        print("Found {} raw data files for {} observation type".format(len(o[k]), k))

    rso_files = o['rso']

    synced_files = get_synced_files(rso_files)

    print()
    for k in synced_files.keys():
        print('{} files where found {}'.format(len(synced_files[k]), k))

    print('\nThe following files were found on remote only (and should be backed up)')
    for raw_file in synced_files['remote_only']:
        print(raw_file)


if __name__ == '__main__':
    observations = get_observations()

    # Check which files are synced locally, and which need to be backed up
    output_birales_stats(observations)

    # get the detection observations
    target_obs_df = get_target_observations(observations["rso"])

    print(target_obs_df.sort_values(by='date', ascending=True))

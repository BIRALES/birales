import os

FILES = ['data.merged.tree', 'data2017.tree']
EXT = '_raw.dat'
LOCAL_DATA_DIR = '/media/denis/backup/birales/'
REMOTE_DATA_DIRS = ['/data/birales/', '/data2/birales/', '/data2/']
observations = {'cas': [], 'cyg': [], 'tau': [], 'vir': [], 'rso': [], 'test': []}


def get_obs_type(file_str):
    for s in observations.keys():
        if s in file_str.lower():
            return s
    return 'rso'


def get_synced_files(rso_files):
    synced_files = {
        'local_and_remote': [],
        'remote_only': [],
        'local_only': []
    }

    for raw_rso_file in rso_files:
        for remote_root in REMOTE_DATA_DIRS:
            if raw_rso_file.startswith(remote_root):
                truncated_raw_rso_file = raw_rso_file[len(remote_root):]

                if os.path.exists(os.path.join(LOCAL_DATA_DIR, truncated_raw_rso_file)):
                    synced_files['local_and_remote'].append(raw_rso_file)
                else:
                    synced_files['remote_only'].append(raw_rso_file)
                break
        else:
            synced_files['local_only'].append(raw_rso_file)

    return synced_files


if __name__ == '__main__':
    rso_campaign = []
    calib_campaign = {}
    for _file in FILES:
        with open(_file, 'rb') as f:
            for line in f:
                line = line.rstrip()  # strip out all tailing whitespace
                if not line.endswith(EXT):
                    continue

                obs_type = get_obs_type(line)
                observations[obs_type].append(line)

    for k in observations.keys():
        print "Found {} raw data files for {} observation type".format(len(observations[k]), k)

    rso_files = observations['rso']

    synced_files = get_synced_files(rso_files)

    print
    for k in synced_files.keys():
        print '{} files where found {}'.format(len(synced_files[k]), k)

    print '\nThe following files were found on remote only (and should be backed up)'
    for raw_file in synced_files['remote_only']:
        print os.path.basename(raw_file)

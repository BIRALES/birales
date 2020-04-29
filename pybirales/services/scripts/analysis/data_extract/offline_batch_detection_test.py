import datetime
import os

from data_extract import get_target_observations, get_observations, get_synced_files
from pybirales.base.observation_manager import ObservationManager
from pybirales.services.scheduler.observation import ScheduledObservation


def get_observation(config_file, pipeline_name, obs_name, raw_filepath):
    parameters = {
        'rawdatareader': {
            'filepath': raw_filepath
        },
        'observation': {
            'name': obs_name
        },
        'manager': {
            'offline': True
        },
        'duration': 7200
    }

    return ScheduledObservation(name=obs_name,
                                pipeline_name=pipeline_name,
                                config_file=config_file,
                                config_parameters=parameters)


def run_observations(config_files, pipeline_name, start_date, end_date):
    om = ObservationManager()

    observations = get_observations()
    # get the detection observations

    synced = get_synced_files(observations['rso'])
    target_obs_df = get_target_observations(synced['all_local'])

    # print target_obs_df.sort_values(by='date', ascending=True)

    date_from = datetime.datetime.strptime(start_date, '%d/%m/%Y')
    date_to = datetime.datetime.strptime(end_date, '%d/%m/%Y')
    mask = (target_obs_df['date'] >= date_from) & (target_obs_df['date'] <= date_to)
    target_obs = target_obs_df[mask]

    for index, row in target_obs.iterrows():
        obs_name = '{}_offline'.format(row.obs_name)
        observation = get_observation(config_files, pipeline_name, obs_name=obs_name, raw_filepath=row.raw_filepath)
        print "Running observation {} from raw file path: {}".format(obs_name, row.raw_filepath)
        om.run(observation)
        break


if __name__ == '__main__':
    CONFIG_ROOT = '/home/denis/.birales/configuration/'
    CONFIG = [os.path.join(CONFIG_ROOT, 'birales.ini'), os.path.join(CONFIG_ROOT, 'detection.ini')]
    PIPELINE = 'msds_detection_pipeline'

    CAMPAIGNS = [
        ('28/03/2018', '29/03/2018'),
        # ('27/02/2019', '05/03/2019'),
        # ('01/04/2019', '10/04/2019'),
        # ('30/07/2019', '25/08/2019')
    ]

    CAMPAIGNS = [
        ('05/03/2019', '05/03/2019'),
    ]

    for campaign in CAMPAIGNS:
        run_observations(CONFIG, PIPELINE, start_date=campaign[0], end_date=campaign[1])

import datetime
import os
import time

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


def _date(date):
    return datetime.datetime.strptime(date, '%d/%m/%Y')


def run_observations(config_files, pipeline_name, start_date, end_date):
    om = ObservationManager()

    observations = get_observations()
    synced = get_synced_files(observations['rso'])
    target_obs_df = get_target_observations(synced['all_local'])

    target_obs = target_obs_df.query("'{}' <= date <= '{}'".format(_date(start_date), _date(end_date)))
    # target_obs = target_obs_df.query(
    #     "(date=='29/03/2018' | date=='03/05/2019') & (obs_name=='tiangong1' | obs_name=='norad_1328')")
    #
    # target_obs = target_obs_df.query(
    #     "(date=='29/03/2018' ) & (obs_name=='tiangong1'  )")

    # target_obs = target_obs_df.query(
    #     "(date=='03/05/2019') & (obs_name=='norad_1328')")
    confirmed = ['norad_16864',
                 'norad_4071',
                 'norad_37820',
                 'norad_25876',
                 'norad_31114',
                 'norad_41765',
                 'norad_1328',
                 'norad_27944',
                 'norad_41765',
                 'norad_1328',
                 'norad_41182',
                 'norad_80155',
                 'norad_2874',
                 'tiangong1']

    target_obs = target_obs[target_obs['obs_name'].isin(confirmed)]

    for index, row in target_obs.iterrows():
        obs_name = '{}_offline'.format(row.obs_name)
        print "Running observation {} from raw file path: {}".format(obs_name, row.raw_filepath)

        # if row.date < datetime.datetime.strptime('01/09/2018', '%d/%m/%Y'):
        #     config_files.append(os.path.join(CONFIG_ROOT, 'detection_old.ini'))

        observation = get_observation(config_files, pipeline_name, obs_name=obs_name, raw_filepath=row.raw_filepath)

        om.run(observation)

        time.sleep(5)


if __name__ == '__main__':
    CONFIG_ROOT = '/home/denis/.birales/configuration/'
    CONFIG = [os.path.join(CONFIG_ROOT, 'birales.ini'), os.path.join(CONFIG_ROOT, 'detection.ini')]
    PIPELINE = 'msds_detection_pipeline'

    CAMPAIGNS = [
        ('28/03/2018', '29/03/2018'),
        ('27/02/2019', '05/03/2019'),
        ('01/04/2019', '10/04/2019'),
        # ('30/07/2019', '25/08/2019')
    ]

    # CAMPAIGNS = [
    #     ('28/03/2018', '29/03/2018'),
    #     ('27/02/2019', '05/03/2019'),
    #     ('01/04/2019', '06/04/2019'),
    #     # ('30/07/2019', '25/08/2019')
    # ]

    # CAMPAIGNS = [
    #     ('28/03/2018', '29/03/2018'),
    # ]

    for campaign in CAMPAIGNS:
        run_observations(CONFIG, PIPELINE, start_date=campaign[0], end_date=campaign[1])

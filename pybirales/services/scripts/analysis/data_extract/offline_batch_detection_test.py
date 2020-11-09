import datetime
import os
import time

from data_extract import get_target_observations, get_observations, get_synced_files
from pybirales.base.observation_manager import ObservationManager
from pybirales.services.scheduler.observation import ScheduledObservation


def get_observation(config_file, pipeline_name, obs_name, raw_filepath, skip=0):
    parameters = {
        'rawdatareader': {
            'filepath': raw_filepath,
            'skip': skip
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


if __name__ == '__main__':
    CONFIG_ROOT = '/home/denis/.birales/configuration/'
    CONFIG = [os.path.join(CONFIG_ROOT, 'birales.ini'), os.path.join(CONFIG_ROOT, 'detection.ini')]
    PIPELINE = 'msds_detection_pipeline'

    # CAMPAIGNS = [
    #     ('28/03/2018', '29/03/2018', None),
    #     ('27/02/2019', '05/03/2019', None),
    #     ('01/04/2019', '06/04/2019', None),
    #     # ('30/07/2019', '25/08/2019', None)
    # ]

    CAMPAIGNS = [
        ('28/03/2018', '29/03/2018',
         [('norad_16864', 0),
          ('norad_4071', 0),
          ('tiangong1', 15),
          ('norad_25876', 0),
          ('norad_31114', 0)]),
        ('27/02/2019', '05/03/2019',
         [('norad_27944', 60),
          ('norad_41765', 35),
          ('norad_1328', 0),
          ('norad_41182', 50)
          ]),
        ('01/04/2019', '10/04/2019',
         [('norad_80155', 39),
          ('norad_2874', 28)
          ]),
    ]

    CAMPAIGNS = [
        # ('29/03/2018', '29/03/2018', [('tiangong1', 15)]),
        # ('05/03/2019', '05/03/2019', [('norad_1328', 50)]),
        # ('29/03/2018', '29/03/2018', [('norad_31114', 0)]),
        # ('27/02/2019', '27/02/2019', [('norad_41765', 58)]),
        # ('05/03/2019', '05/03/2019', [('norad_41765', 35)]),
        # ('27/02/2019', '27/02/2019', [('norad_27944', 60)]),
        # ('05/04/2019', '05/04/2019', [('norad_2874', 30)]),
        # ('29/03/2018', '29/03/2018', [('norad_25876', 0)]),
        ('27/02/2019', '27/02/2019', [('norad_1328', 10)]),
        # ('05/03/2019', '05/03/2019', [('norad_41182', 50)]),
        # ('29/03/2018', '29/03/2018', [('norad_16864', 0)]),
        # ('29/03/2018', '29/03/2018', [('norad_4071', 0)]),
        # ('29/03/2018', '29/03/2018', [('norad_37820', 0)]),
        # ('29/03/2018', '29/03/2018', [('norad_31114', 0)]),

    ]

    om = ObservationManager()
    observations = get_observations()
    synced = get_synced_files(observations['rso'])
    target_obs_df = get_target_observations(synced['all_local'])

    for campaign_start, campaign_end, targets in CAMPAIGNS:
        target_obs = target_obs_df.query("'{}' <= date <= '{}'".format(_date(campaign_start), _date(campaign_end)))

        if not targets:
            continue

        for target_name, skip in targets:
            target_obs1 = target_obs[target_obs['obs_name'] == target_name]

            for index, row in target_obs1.iterrows():
                obs_name = '{}_offline'.format(row.obs_name)
                print("Running observation {} from raw file path: {}".format(obs_name, row.raw_filepath))

                observation = get_observation(CONFIG, PIPELINE, obs_name=obs_name, raw_filepath=row.raw_filepath,
                                              skip=skip)

                om.run(observation)

                time.sleep(5)

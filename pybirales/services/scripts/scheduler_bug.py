import datetime
import json
import os
import threading
import time

from pybirales.base.observation_manager import ObservationManager
from pybirales.birales import BiralesFacade
from pybirales.birales_config import BiralesConfig
from pybirales.repository.message_broker import broker
from pybirales.services.scheduler.observation import ScheduledObservation
from pybirales.services.scheduler.scheduler import ObservationsScheduler


def get_observation(config_file, pipeline_name, obs_name, raw_filepath, duration=60 * 2):
    parameters = {
        'rawdatareader': {
            'filepath': raw_filepath,
            'skip': 0
        },
        'observation': {
            'name': obs_name
        },
        'manager': {
            'offline': True
        },
        'duration': duration
    }

    return ScheduledObservation(name=obs_name,
                                pipeline_name=pipeline_name,
                                config_file=config_file,
                                config_parameters=parameters)


def test_1():
    CONFIG_ROOT = '/home/denis/.birales/configuration/'
    CONFIG = [os.path.join(CONFIG_ROOT, 'birales.ini'), os.path.join(CONFIG_ROOT, 'detection.ini')]
    PIPELINE = 'msds_detection_pipeline'
    ROOT = '/media/denis/backup/birales'

    om = ObservationManager()
    os = ObservationsScheduler()

    observations = [
        ("norad_1328", ROOT + "/2019/2019_03_05/norad_1328/norad_1328_raw.dat"),
        ("norad_41182", ROOT + "/2019/2019_03_05/norad_41182/norad_41182_raw.dat")
    ]

    for obs in observations:
        obs_name, raw_filepath = obs
        observation = get_observation(CONFIG, PIPELINE, obs_name=obs_name, raw_filepath=raw_filepath)

        om.run(observation)

        time.sleep(5)


def mock_observation(name, delay, duration):
    time.sleep(1)

    root_dir = os.path.join(os.environ['HOME'], '.birales')
    config_parameters = {
        "beamformer": {
            "reference_declination": float(45.)
        },
        "observation": {
            "name": 'T2',
            'transmitter_frequency': 410.085
        },
        "start_time": '{:%Y-%m-%d %H:%M:%S}Z'.format(datetime.datetime.utcnow() + datetime.timedelta(seconds=delay)),
        "duration": duration
    }

    data = json.dumps({
        "name": name,
        "type": 'observation',
        "pipeline": 'detection_pipeline',
        "config_file": [
            os.path.join(root_dir, "configuration/birales.ini"),
            os.path.join(root_dir, "configuration/detection.ini"),
        ],
        "config_parameters": config_parameters
    })

    return data


def app_mock_worker():
    delay = 10
    print(f"Will start to schedule observations in {delay} seconds")
    time.sleep(delay)

    broker.publish(b'birales_scheduled_obs', mock_observation("Observation_1", delay=10, duration=10))

    broker.publish(b'birales_scheduled_obs', mock_observation("Observation_2", delay=30, duration=10))


if __name__ == '__main__':
    CONFIG_ROOT = '/home/denis/.birales/configuration/'

    app_thread = threading.Thread(target=app_mock_worker, name='App. Mock')

    # Load the BIRALES configuration from file
    config = BiralesConfig([CONFIG_ROOT + 'birales.ini'], {})

    config.load()

    # Initialise the BIRALES Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Start a thread that mocks the frontend application
    app_thread.start()

    # Start the BIRALES scheduler
    bf.start_scheduler(schedule_file_path=None, file_format='json')

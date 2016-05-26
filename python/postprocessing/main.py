from controllers.SpaceDebrisController import SpaceDebrisController
import config.log as log_config
import config.application as config
import cProfile
import pstats
import StringIO
import logging as log
import os
import sys
import getopt
from glob import glob

log.basicConfig(format=log_config.FORMAT, level=log.DEBUG)


# todo - Separate viewing of results from postprocessing of beam data
# todo - do not commit vendor files


def run():
    """
    Post process the data
    :return:
    """

    myopts, args = getopt.getopt(sys.argv[1:], "o:d:")

    observation = None
    data_set = None
    for o, a in myopts:
        if o is '-d':
            data_set = a
        elif o is '-o':
            observation = o

    def get_observations():
        data = os.listdir(config.DATA_FILE_PATH)
        observations_dir = [os.path.join(config.DATA_FILE_PATH, obs) for obs in data if
                            os.path.isdir(os.path.join(config.DATA_FILE_PATH, obs))]
        return observations_dir

    def get_data_sets(observation_dir):
        data = os.listdir(observation_dir)
        data_sets_dirs = [os.path.join(observation_dir, d) for d in data if
                          os.path.isdir(os.path.join(observation_dir, d))]
        return data_sets_dirs

    if observation is None and data_set is None:
        # observations = [os.path.join(config.DATA_FILE_PATH, o) for o in os.listdir(config.DATA_FILE_PATH) if
        #                 os.path.isdir(os.path.join(config.DATA_FILE_PATH, o))]
        # for observation in observations:
        #     if os.path.isdir(os.path.join(config.DATA_FILE_PATH, observation)):
        #         for data_set in os.listdir(os.path.join(config.DATA_FILE_PATH, observation)):
        #             if os.path.isdir(os.path.join(config.DATA_FILE_PATH, observation, data_set)):
        #                 odc = SpaceDebrisController(observation=observation, data_set=data_set)
        #                 odc.run()

        observations = get_observations()
        for observation_dir in observations:
            data_sets = get_data_sets(observation_dir)
            for data_set_dir in data_sets:
                observation = os.path.basename(observation_dir)
                data_set = os.path.basename(data_set_dir)
                odc = SpaceDebrisController(observation=observation, data_set=data_set)
                odc.run()

    else:
        odc = SpaceDebrisController(observation=observation, data_set=data_set)
        odc.run()

        # else:
        #     # The observations / data sets which will be processed
        #     observations = {
        #         'medicina_07_03_2016': [
        #             'mock_1358',
        #             '1358',
        #             '24773',
        #             '25484',
        #             '40058',
        #             '5438',
        #             '7434'
        #         ]
        #     }


if not config.PROFILE:
    run()
else:
    cProfile.run('run()', config.PROFILE_LOG_FILE)
    stream = StringIO.StringIO()
    pr = cProfile.Profile()
    stats = pstats.Stats(config.PROFILE_LOG_FILE, stream=stream).sort_stats('cumulative')
    stats.print_stats()
    print stream.getvalue()

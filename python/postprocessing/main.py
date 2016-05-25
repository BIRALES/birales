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

    if observation is None and data_set is None:
        for observation in os.listdir(config.DATA_FILE_PATH):
            for data_set in os.listdir(os.path.join(config.DATA_FILE_PATH, observation)):
                odc = SpaceDebrisController(observation=observation, data_set=data_set, tx=399)
                odc.run()
    else:
        odc = SpaceDebrisController(observation=observation, data_set=data_set, tx=399)
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

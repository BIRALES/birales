from app.sandbox.postprocessing.controllers.SpaceDebrisController import SpaceDebrisController
import app.sandbox.postprocessing.config.log as log_config
import app.sandbox.postprocessing.config.application as config
import cProfile
import pstats
import StringIO
import logging as log

log.basicConfig(format = log_config.FORMAT, level = log.DEBUG)


def run():
    observations = {
        'medicina_07_03_2016': [
            # 'mock_1358',
            # '1358',
            '24773',
            '25484',
            '40058',
            '5438',
            '7434'
        ]
    }

    for observation, data_sets in observations.iteritems():
        for data_set in data_sets:
            odc = SpaceDebrisController(observation = observation, data_set = data_set, tx = 399)
            odc.run()


if not config.PROFILE:
    run()
else:
    cProfile.run('run()', config.PROFILE_LOG_FILE)
    stream = StringIO.StringIO()
    pr = cProfile.Profile()
    stats = pstats.Stats(config.PROFILE_LOG_FILE, stream = stream).sort_stats('cumulative')
    stats.print_stats()
    print stream.getvalue()

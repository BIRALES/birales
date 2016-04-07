from app.sandbox.postprocessing.controllers.SpaceDebrisController import SpaceDebrisController

import cProfile
import pstats
import StringIO
import logging
import logstash

test_logger = logging.getLogger('python-logstash-logger')
test_logger.setLevel(logging.INFO)
test_logger.addHandler(logstash.LogstashHandler('localhost', 5959, version=1))

pr = cProfile.Profile()
pr.enable()

# test_logger.info('python-logstash: Space Debris Controller Started')
odc = SpaceDebrisController()
odc.run()

pr.disable()
s = StringIO.StringIO()
ps = pstats.Stats(pr, stream = s).sort_stats('cumulative')
ps.print_stats()
# print s.getvalue()

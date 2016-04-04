from app.sandbox.postprocessing.controllers.SpaceDebrisController import SpaceDebrisController

import cProfile
import pstats
import StringIO

pr = cProfile.Profile()
pr.enable()

odc = SpaceDebrisController()
odc.run()

pr.disable()
s = StringIO.StringIO()
ps = pstats.Stats(pr, stream = s).sort_stats('cumulative')
ps.print_stats()
print s.getvalue()

from OrbitDeterminationInputGenerator import OrbitDeterminationInputGenerator
from BeamData import BeamData
from SpaceDebrisController import SpaceDebrisController

# od = OrbitDeterminationInputGenerator()
# bd = BeamData()
#
# # bd.visualise(power, time, freq, 'Mock_BeamData')
#
# hough = od.line_detection(bd)


odc = SpaceDebrisController()
odc.run()

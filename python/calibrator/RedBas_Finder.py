import numpy as np
from Calibrator import MSReader
from Calibrator import RedBasFinder


## Initialize MS Reader
MSReader = MSReader()

### Obtain RedBas from MS

user_defined_name = 'BEST_2'
ms_name = 'I_O/' + user_defined_name + '.ms'

## Read MS
np.set_printoptions(threshold='nan')
tb.open(ms_name, nomodify=False)
uvw = tb.getcol("UVW")
UVW = np.array(uvw.transpose())
antenna1 = np.array(tb.getcol("ANTENNA1"))
antenna2 = np.array(tb.getcol("ANTENNA2"))

## Parse to MS Reader
#MSReader = MSReader()
MSReader.read(UVW, antenna1, antenna2)

## Initialize RedBas Finder with UVWs
RBFinder = RedBasFinder(UVW, antenna1, antenna2)

## Find Redundant Baselines and Parse to txt
RBFinder.find()
filename = 'I_O/redbas.txt'
RBFinder.parse(filename)

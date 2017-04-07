import numpy as np
import h5py
from Calibrator import RBCal
from datetime import datetime

user_defined_name = 'BEST_2'
h5file = 'I_O/' + user_defined_name + '.h5' 
gainfile = 'I_O/' + user_defined_name + '_gain.txt'
phasefile = 'I_O/' + user_defined_name + '_phase.txt'
RBname = 'I_O/redbas.txt'

## Initialize RBCal
rbCal = RBCal()
rbCal.load_RBs(RBname)

## Launch RBCal
rbCal.calib(h5file)

## Save Coeffs to File
rbCal.parse_coeff(gainfile, phasefile)


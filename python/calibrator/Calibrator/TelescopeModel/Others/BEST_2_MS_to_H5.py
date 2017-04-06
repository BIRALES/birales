import numpy as np
import math 
import os
import h5py

np.set_printoptions(threshold='nan')
tb.open("BEST_2.ms", nomodify=False)
uvw = tb.getcol("UVW")
uvw_col = np.array(uvw.transpose())
antenna1 = np.array(tb.getcol("ANTENNA1"))
antenna2 = np.array(tb.getcol("ANTENNA2"))
dataget = np.array(tb.getcol("DATA"))

npols = dataget.shape[0]
nchans = dataget.shape[1]
ncross = dataget.shape[2]

nantennas = np.array([antenna1, antenna2])

f = h5py.File("BEST_2.h5", "w")
aset = f.create_dataset("Basl_Indices", (nantennas.shape[0], nantennas.shape[1]))
dset = f.create_dataset("Visibilities", (npols,nchans,ncross), dtype = 'c16')
aset[:,:] = nantennas[:,:]
dset[:,:,:] = dataget[:,:,:]
f.flush()
f.close()












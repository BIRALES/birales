import numpy as np
import h5py

class Correlator:

    def __init__(self):

        self.nchans = None
        self.npol = None
        self.nants = None
        self.nsamp = None
        self.data = None
        self.nbasl = None
        self.ncross = None
        self.corr_m = None
        self.stoke_m = None
        self.timestep = None
        self.antenna1 = []
        self.antenna2 = []
        self.basl_no = []

    def raw_feed_reader(self, raw_data, time, obstime):

        self.nchans = len(raw_data[:,0,0,0])
        self.npol = len(raw_data[0,:,0,0])
        self.nants = len(raw_data[0,0,:,0])
        self.nsamp = len(raw_data[0,0,0,:])     
        self.data = raw_data
        self.time = time
        self.obstime = obstime
        self.dset = None

    def define_parameters(self):

        self.nbasl = int(0.5*((self.nants**2) - self.nants))
        self.ncross = self.npol**2

    def run_cross_corr(self):

        self.corr_m = np.zeros((self.nchans, self.nbasl, self.ncross), dtype=np.complex)
        
        if self.npol == 1:
            for c in range(self.nchans):
	        cr = 0
	        for a in range(self.nants):
		    for a2 in range(a+1, self.nants):
                        self.corr_m[c,cr,0] = np.dot(self.data[c,0,a,:], np.conj(self.data[c,0,a2,:]))
                        if c == 0:
                            self.antenna1.append(a)
                            self.antenna2.append(a2)
                            self.basl_no.append(cr)
                        cr += 1

        if self.npol == 2:
            for c in range(self.nchans):
	        cr = 0
	        for a in range(self.nants):
		    for a2 in range(a+1, self.nants):
                        self.corr_m[c,cr,0] = np.dot(self.data[c,0,a,:], np.conj(self.data[c,0,a2,:]))
                        self.corr_m[c,cr,1] = np.dot(self.data[c,0,a,:], np.conj(self.data[c,1,a2,:]))
                        self.corr_m[c,cr,2] = np.dot(self.data[c,1,a,:], np.conj(self.data[c,0,a2,:]))
                        self.corr_m[c,cr,3] = np.dot(self.data[c,1,a,:], np.conj(self.data[c,1,a2,:]))
                        if c == 0:
                            self.antenna1.append(a)
                            self.antenna2.append(a2)
                            self.basl_no.append(cr)
                        cr += 1

    def generate_stokes(self):

        self.stoke_m = np.zeros((self.nchans, self.nbasl, self.ncross))

        for c in range(self.nchans):
	    for a in range(self.nbasl):
                self.stoke_m[c,a,0] = 0.5 * (self.corr_m[c,a,0] + self.corr_m[c,a,3])
                self.stoke_m[c,a,1] = 0.5 * (self.corr_m[c,a,0] - self.corr_m[c,a,3])
                self.stoke_m[c,a,2] = 0.5 * (self.corr_m[c,a,1] + self.corr_m[c,a,2])
                self.stoke_m[c,a,3] = -0.5 * np.imag(self.corr_m[c,a,1] - self.corr_m[c,a,2])

    def parse_corr_to_h5(self, filename):

        if self.time == 0:
            f = h5py.File(filename, "w")
            name = "Vis"
            dset = f.create_dataset(name, (self.obstime, self.nchans, self.nbasl, self.ncross), dtype='c16') 
            dset[self.time,:,:,:] = self.corr_m[:,:,:]
            dset2 = f.create_dataset("Baselines", (len(self.antenna2), 3))
            dset2[:,:] = np.transpose([self.basl_no, self.antenna1, self.antenna2])
            f.flush()
            f.close()

        if self.time > 0:
            f = h5py.File(filename, "a")
            dset = f["Vis"]
            dset[self.time,:,:,:] = self.corr_m[:,:,:]
            f.flush()
            f.close()

    def parse_stokes_to_h5(self, filename):

        f2 = h5py.File(filename, "w")
        dset3 = f2.create_dataset("Stokes", (self.nchans, self.nbasl, self.ncross))
        dset3[:,:,:] = self.stoke_m[:,:,:]
        dset4 = f2.create_dataset("Baselines", (len(self.antenna2), 3))
        dset4[:,:] = np.transpose([self.basl_no, self.antenna1, self.antenna2])
        f2.flush()
        f2.close()

 
			



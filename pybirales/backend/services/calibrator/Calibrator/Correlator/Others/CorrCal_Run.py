import numpy as np
import h5py
from Calibrator import Correlator
from datetime import datetime



### Receive Antenna Raw Feed to Correlator and Generate Corr Matrix

## Raw Data in and User-Defined Name
raw_data = np.ones((1, 1, 32, 70000), dtype=np.complex)
user_defined_name = 'BEST_2'

## Initialize Correlator
Corr = Correlator()
h5file = 'I_O/' + user_defined_name + '.h5' 

time = 0
obs_time = 60

start_obs = datetime.now()

for i in range(obs_time):
    
    start = datetime.now()

    ## Receive Raw Feed
    Corr.raw_feed_reader(raw_data, time, obs_time)

    if i == 0:
        ## Set Up Correlator
        Corr.define_parameters()

    ## Run Correlation
    Corr.run_cross_corr()

    ## Parse Corr Matrix to H5
    Corr.parse_corr_to_h5(h5file)

    time += 1

    end = datetime.now()
    print(end - start)

end_obs = datetime.now()
print(end_obs - start_obs)


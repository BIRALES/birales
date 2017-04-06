import numpy as np
from Correlator import Correlator
from UVW import UVW

antennas = [[-34.8875321, 6.804708, 0],
                    [-34.8875321, 34.023538, 0],
                    [-34.8875321, 61.242368, 0],
                    [-34.8875321, 88.461198, 0],
                    [-34.8875321, 20.414123, 0],
                    [-34.8875321, 47.632953, 0],
                    [-34.8875321, 74.851783, 0],
                    [-34.8875321, 102.070613, 0],
                    [-26.8875321, 6.804708, 0],
                    [-26.8875321, 34.023538, 0],
                    [-26.8875321, 61.242368, 0],
                    [-26.8875321, 88.461198, 0],
                    [-26.8875321, 20.414123, 0],
                    [-26.8875321, 47.632953, 0],
                    [-26.8875321, 74.851783, 0],
                    [-26.8875321, 102.070613, 0],
                    [-18.8875321, 6.804708, 0],
                    [-18.8875321, 34.023538, 0],
                    [-18.8875321, 61.242368, 0],
                    [-18.8875321, 88.461198, 0],
                    [-18.8875321, 20.414123, 0],
                    [-18.8875321, 47.632953, 0],
                    [-18.8875321, 74.851783, 0],
                    [-18.8875321, 102.070613, 0],
                    [-10.8875321, 6.804708, 0],
                    [-10.8875321, 34.023538, 0],
                    [-10.8875321, 61.242368, 0],
                    [-10.8875321, 88.461198, 0],
                    [-10.8875321, 20.414123, 0],
                    [-10.8875321, 47.632953, 0],
                    [-10.8875321, 74.851783, 0],
                    [-10.8875321, 102.070613, 0]]

## Get_UVW_from_XYZ
UVW = UVW()

# Parse Antennas and Create UVW array
UVW.antenna_parser(antennas)
UVW.XYZ_to_UVW()

print(UVW.baselines)


## Run_Correlator
# Generate mock data in required format: (nchan, npol, nants, nsamp) 
raw_data = np.ones((1, 1, 32, 70000))

# Initialize Correlator
Corr = Correlator()

# Input Raw Antenna Feed to Reader and Prepare Parameters for Corr
Corr.raw_feed_reader(raw_data)
Corr.define_parameters()

# Run Cross Correlator and Generate Stokes Matrix
Corr.run_cross_corr()
#Corr.generate_stokes()


# Save to HDF5 format
corr_m_filename = "Visibilities.h5"
Corr.parse_corr_to_h5(corr_m_filename)

stokes_filename = "Stokes.h5"
#Corr.parse_stokes_to_h5(stokes_filename)

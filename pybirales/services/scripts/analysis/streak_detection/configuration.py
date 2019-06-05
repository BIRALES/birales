import numpy as np

ROOT = "/home/denis/.birales/visualisation/fits"
OUT_DIR = "/home/denis/.birales/visualisation/analysis"

OBS_NAME = "NORAD_1328"
N_TRACKS = 20
TD = 262144 / 78125 / 32.
CD = 78125 / 8192.
F = (1. / TD) / (1. / CD)
GRADIENT_RANGE = np.array([-0.57, -50.47]) / F
TRACK_LENGTH_RANGE = np.array([5, 15]) / TD  # in seconds
TRACK_THICKNESS = 1
# FITS_FILE = "norad_1328/norad_1328_raw_0.fits"
FITS_FILE = "filter_test/filter_test_raw_0.fits"
FITS_FILE = "Observation_2019-05-17T1202/Observation_2019-05-17T1202_raw_1.fits"
VISUALISE = True
SAVE_FIGURES = False
SEED = 56789
np.random.seed(SEED)

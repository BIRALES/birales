import os.path

# Directory where observation data is stored
DATA_FILE_PATH = 'data/'

# Directory where to store results
RESULTS_FILE_PATH = 'public/results'

# Threshold at which a detection is determined
SNR_DETECTION_THRESHOLD = 1.0

# Maximum number of detections per beam
MAX_DETECTIONS = 3

# Algorithm to use for Detection [LineSpaceDebrisDetectionStrategy]
DETECTION_STRATEGY = 'LineSpaceDebrisDetectionStrategy'

# Environment [development, staging, production]
ENVIRONMENT = 'development'

# How verbose the logging is [0, 1, 2]
LOG_LEVEL = 0

# Persist results
PERSIST_RESULTS = True

# Observation name
OBSERVATION_NAME = 'medicina_07_03_2016'

# Data-set to analyse
DATA_SET = '24773'

# File name to give to the unprocessed input beam
INPUT_BEAM_FILE_NAME = 'input_beam'

# File name to give to the filtered input beam
FILTERED_BEAM_FILE_NAME = 'filtered_beam'

# File name to give to the detections in the input beam
DETECTIONS_BEAM_FILE_NAME = 'detection_profile'

# File name to give to the detections in the input beam
OD_FILE_NAME = 'orbit_determination_data'

# Inferred config variables
OBSERVATION_DATA_DIR = os.path.join(DATA_FILE_PATH, OBSERVATION_NAME, DATA_SET)

# Output directory for beams
BEAM_OUTPUT_DATA = os.path.join(RESULTS_FILE_PATH, OBSERVATION_NAME, DATA_SET, 'beams')

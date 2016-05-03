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

# Visualise input data (Heat map)
SAVE_INPUT_DATA = False

# Save candidates parameters to disk (Html Table or Database)
SAVE_CANDIDATES = True

# Visualise candidates on beam data (Heat map)
VIEW_CANDIDATES = True

# File name to give to the unprocessed input beam
INPUT_BEAM_FILE_NAME = 'input_beam'

# File name to give to the filtered input beam
FILTERED_BEAM_FILE_NAME = 'filtered_beam'

# File name to give to the detections in the input beam
DETECTIONS_BEAM_FILE_NAME = 'detection_profile'

# File name to give to the detections in the input beam
OD_FILE_NAME = 'orbit_determination_data'

# Output format for visualisations [HTML, PNG]
VIEW_OUTPUT_FORMAT = 'PNG'

# Location where to save the database [HTML, DB]
CANDIDATES_SAVE = 'DB'

# View candidates (use for debug purposes)
DEBUG_CANDIDATES = False

# Profile application
PROFILE = False

# Profile log file
PROFILE_LOG_FILE = 'log/profile.log'

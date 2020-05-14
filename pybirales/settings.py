"""

Settings file will be overwritten on run-time. Properties below are
set to None so as to remove warnings in source code.


"""

channeliser = None
beamformer = None
persister = None
rawpersister = None
rawdatareader = None
channelplotter = None
bandpassplotter = None
antennaplotter = None
persisters = None
corrmatrixpersister = None
correlator = None
terminator = None
manager = None
calibration = None
database = None
instrument = None
logger_root = None
fits_persister = None
scheduler = None
# handler_file_handler = None
roach_config_files = None
feng_configuration = None
generator = None
rso_generator = None


class receiver:
    backend_config_filepath = 'configuration/backend/roach_backend.ini'
    force_start = True


class observation:
    id = -1


class detection:
    similarity_thold = 0.1
    linearity_thold = 0.95
    gradient_thold = [-57, -291.47]
    doppler_range = [-19688, 19507]
    debug_candidates = False
    save_tdm = False
    enable_gradient_thold = True


class manager:
    profile_timeit = False

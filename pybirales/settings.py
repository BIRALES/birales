"""

Settings file will be overwritten on run-time. Properties below are
set to None so as to remove warnings in source code.


"""
digital_backend = None
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

terminator = None
calibration = None
database = None
logger_root = None
fits_persister = None
scheduler = None
# handler_file_handler = None
roach_config_files = None
feng_configuration = None
generator = None
rso_generator = None
tpm_receiver = None


class instrument:
    name_prefix = '1N'


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


class correlator:
    channel_start = None
    channel_end = None

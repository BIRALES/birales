"""
Settings file will be overwritten on run-time. Properties below are
set to None to remove warnings in source code.


"""
digital_backend = None
channeliser = None
beamformer = None
persister = None
rawpersister = None
rawdatareader = None
persisters = None
corrmatrixpersister = None

terminator = None
calibration = None
database = None
logger_root = None
fits_persister = None
scheduler = None
generator = None
rso_generator = None
tpm_receiver = None


class instrument:
    name_prefix = '1N'


class observation:
    id = -1


class detection:
    similarity_thold = 0.1
    linearity_thold = 0.95
    gradient_thold = [-57, -291.47]
    doppler_range = [-19688, 19507]
    min_beams = 2
    min_missing_score = 0.25
    debug_candidates = False
    save_tdm = False
    enable_gradient_thold = True
    algorithm = 'msds'


class manager:
    use_gpu = False
    gpu_device_id = 0


class correlator:
    channel_start = None
    channel_end = None

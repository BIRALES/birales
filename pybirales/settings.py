"""

Settings file will be overwritten on run-time. Properties below are
set to None so as to remove warnings in source code.


"""

channeliser = None
beamformer = None
detection = None
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
observation = None
calibration = None
database = None
instrument = None
logger_root = None
fits_persister = None
scheduler = None
# handler_file_handler = None
roach_config_files = None
feng_configuration = None

class receiver:
    backend_config_filepath = 'configuration/backend/roach_backend.ini'
    force_start = True
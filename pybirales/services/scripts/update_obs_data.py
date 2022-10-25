import pickle
from bson.objectid import ObjectId
import datetime


obs_info_str = {
    "name": "SURVEY_53240",
    "date_time_start": datetime.datetime(year=2022, month=7, day=28, hour=5, minute=20, second=29, microsecond=0),
    "date_time_end": datetime.datetime(year=2022, month=7, day=28, hour=6, minute=37, second=6, microsecond=668),
    "pipeline": "detection_pipeline",
    "type": "observation",
    "config_parameters": {
        "beamformer": {
            "reference_declination": -12.55
        },
        "observation": {
            "name": "SURVEY_53240",
            "target_name": "53240",
            "transmitter_frequency": 410.085
        },
        "start_time": "2022-07-28 05:20:29+00:00",
        "duration": 4590.0
    },
    "config_file": [
        "/home/oper/.birales/configuration/birales.ini",
        "/home/oper/.birales/configuration/detection.ini"
    ],
    "noise_beams": [
        0.991799801588058,
        0.997706443071365,
        0.982311308383942,
        0.976758748292923,
        0.985466808080673,
        0.99256244301796,
        1.0076305270195,
        0.996696650981903,
        1.00226372480392,
        0.998375505208969,
        0.983832210302353,
        0.964562982320786,
        0.960320681333542,
        0.969720721244812,
        1.00769627094269,
        0.996830761432648,
        0.984899014234543,
        0.992779850959778,
        0.99050372838974,
        0.984800815582275,
        0.958783864974976,
        0.961619257926941,
        0.959451705217361,
        0.972938656806946,
        0.983898431062698,
        0.978157132863998,
        0.984301805496216,
        0.989287406206131,
        0.981103599071503,
        0.985985279083252,
        0.99268627166748,
        0.986374288797379
    ],
    "status": "finished",
    "created_at": datetime.datetime(year=2022, month=7, day=27, hour=12, minute=55, second=59, microsecond=0),
    "principal_created_at": datetime.datetime(year=2022, month=7, day=27, hour=12, minute=55, second=59, microsecond=668),
    "settings": {
        "digital_backend": {
            "configuration_file": "/home/oper/Software/birales/pybirales/configuration/backend/medicina_tpm.yml"
        },
        "channeliser": {
            "nchans": 8192,
            "ntaps": 8,
            "nthreads": 4,
            "use_numba": False
        },
        "beamformer": {
            "nthreads": 8,
            "reference_declination": -12.55,
            "nbeams": 32,
            "pointings": [
                [
                    -3.2,
                    -0.5
                ],
                [
                    -3.2,
                    0.5
                ],
                [
                    -1.6,
                    -2
                ],
                [
                    -1.6,
                    -1.5
                ],
                [
                    -1.6,
                    -1
                ],
                [
                    -1.6,
                    -0.5
                ],
                [
                    -1.6,
                    0
                ],
                [
                    -1.6,
                    0.5
                ],
                [
                    -1.6,
                    1
                ],
                [
                    -1.6,
                    1.5
                ],
                [
                    -1.6,
                    2
                ],
                [
                    0,
                    -2
                ],
                [
                    0,
                    -1.5
                ],
                [
                    0,
                    -1
                ],
                [
                    0,
                    -0.5
                ],
                [
                    0,
                    0
                ],
                [
                    0,
                    0.5
                ],
                [
                    0,
                    1
                ],
                [
                    0,
                    1.5
                ],
                [
                    0,
                    2
                ],
                [
                    1.6,
                    -2
                ],
                [
                    1.6,
                    -1.5
                ],
                [
                    1.6,
                    -1
                ],
                [
                    1.6,
                    -0.5
                ],
                [
                    1.6,
                    0
                ],
                [
                    1.6,
                    0.5
                ],
                [
                    1.6,
                    1
                ],
                [
                    1.6,
                    1.5
                ],
                [
                    1.6,
                    2
                ],
                [
                    3.2,
                    -0.5
                ],
                [
                    3.2,
                    0
                ],
                [
                    3.2,
                    0.5
                ]
            ],
            "reference_antenna_location": [
                11.6459889,
                44.52357778
            ],
            "disable_antennas": [],
            "antenna_locations": [
                [
                    0,
                    0,
                    0
                ],
                [
                    5.6665,
                    0,
                    0
                ],
                [
                    11.333,
                    0,
                    0
                ],
                [
                    16.999,
                    0,
                    0
                ],
                [
                    0,
                    10,
                    -0.01
                ],
                [
                    5.6665,
                    10,
                    -0.01
                ],
                [
                    11.333,
                    10,
                    -0.01
                ],
                [
                    16.999,
                    10,
                    -0.01
                ],
                [
                    0,
                    20,
                    -0.02
                ],
                [
                    5.6665,
                    20,
                    -0.02
                ],
                [
                    11.333,
                    20,
                    -0.02
                ],
                [
                    16.999,
                    20,
                    -0.02
                ],
                [
                    0,
                    30,
                    -0.03
                ],
                [
                    5.6665,
                    30,
                    -0.03
                ],
                [
                    11.333,
                    30,
                    -0.03
                ],
                [
                    16.999,
                    30,
                    -0.03
                ],
                [
                    0,
                    40,
                    -0.04
                ],
                [
                    5.6665,
                    40,
                    -0.04
                ],
                [
                    11.333,
                    40,
                    -0.04
                ],
                [
                    16.999,
                    40,
                    -0.04
                ],
                [
                    0,
                    50,
                    -0.05
                ],
                [
                    5.6665,
                    50,
                    -0.05
                ],
                [
                    11.333,
                    50,
                    -0.05
                ],
                [
                    16.999,
                    50,
                    -0.05
                ],
                [
                    0,
                    60,
                    -0.06
                ],
                [
                    5.6665,
                    60,
                    -0.06
                ],
                [
                    11.333,
                    60,
                    -0.06
                ],
                [
                    16.999,
                    60,
                    -0.06
                ],
                [
                    0,
                    70,
                    -0.07
                ],
                [
                    5.6665,
                    70,
                    -0.07
                ],
                [
                    11.333,
                    70,
                    -0.07
                ],
                [
                    16.999,
                    70,
                    -0.07
                ]
            ],
            "apply_calib_coeffs": True
        },
        "persister": {
            "filename_suffix": "_beam",
            "use_timestamp": False
        },
        "rawpersister": {
            "filename_suffix": "_raw",
            "use_timestamp": False
        },
        "rawdatareader": {
            "filepath": "/mnt/fahal_data/2018_02_06/CasATest3/CasATest3_raw.dat",
            "config_ext": ".pkl",
            "nsamp": 1310720,
            "nants": 32,
            "npols": 1,
            "nsubs": 1,
            "skip": 0
        },
        "channelplotter": {
            "beam_range": 0,
            "nof_samples": 1024
        },
        "bandpassplotter": {
            "beam_range": [
                4,
                5
            ],
            "nof_samples": 1024
        },
        "antennaplotter": {
            "antenna_range": [
                1,
                2
            ],
            "nof_samples": 4096
        },
        "persisters": {
            "directory": "/storage/data/birales/"
        },
        "corrmatrixpersister": {
            "filename_suffix": "_corr",
            "use_timestamp": False,
            "corr_matrix_filepath": False
        },
        "correlator": {
            "integrations": 327680
        },
        "terminator": {},
        "manager": {
            "enable_plotting": False,
            "plot_update_rate": 3,
            "loggging_config_file_path": "configuration/logging.ini",
            "profile": False,
            "profiler_file_path": "/var/log/birales/profiling",
            "profile_timeit": False,
            "detector_enabled": True,
            "offline": False,
            "save_raw": True,
            "save_beam": False,
            "debug": False
        },
        "calibration": {
            "stefcal": False,
            "frequency": 410.109375,
            "integration_time": 3.83479222857143,
            "coeffs_filepath": "coeffs.txt",
            "model_generation": False,
            "test_run_check": False,
            "transit_run": True,
            "calib_check_path": "corr_calib.h5"
        },
        "database": {
            "authentication": True,
            "name": "birales",
            "host": "localhost",
            "port": 27017,
            "user": "birales_rw",
            "password": "rw_Sept03",
            "redis_host": "127.0.0.1",
            "redis_port": 6379
        },
        "logger_root": {
            "level": "DEBUG",
            "handlers": "stream_handler",
            "propagate": 0
        },
        "fits_persister": {
            "visualise_filtered_beams": [],
            "visualise_raw_beams": [],
            "visualise_fits_dir": ".birales/visualisation/fits"
        },
        "scheduler": {
            "auto_calibrate": False
        },
        "generator": {
            "nsamp": 262144,
            "nants": 32,
            "nsubs": 1,
            "complex": True,
            "nbits": 64,
            "npols": 1
        },
        "rso_generator": {
            "nsamp": 262144,
            "nants": 32,
            "nsubs": 1,
            "complex": True,
            "nbits": 64,
            "npols": 1,
            "samples_per_second": 156250,
            "mean_noise_power": 0.00173607422039,
            "rso_freq": 2,
            "doppler_range": [
                -19688,
                19507
            ],
            "doppler_gradient_range": [
                -0.57,
                -2901.47
            ],
            "snr_range": [
                5,
                50
            ],
            "track_length_range": [
                2,
                10
            ],
            "tx_snr": 20
        },
        "tpm_receiver": {
            "nsamp": 1310720,
            "nants": 32,
            "nsubs": 1,
            "nbits": 32,
            "npols": 1,
            "port": 4660,
            "ip": "10.0.10.201"
        },
        "instrument": {
            "enable_pointing": False,
            "name_prefix": "'1N'"
        },
        "receiver": {
            "backend_config_filepath": "configuration/backend/roach_backend.ini",
            "start_time": 1496332160,
            "daq_file_path": "/usr/local/lib/libaavsdaq.so",
            "nsamp": 262144,
            "nants": 32,
            "nsubs": 1,
            "nbits": 64,
            "npols": 1,
            "complex": True,
            "port": 7200,
            "interface": "enp6s0",
            "ip": "192.168.11.11",
            "frame_size": 9000,
            "frames_per_block": 32,
            "nblocks": 256,
            "force_start": True
        },
        "observation": {
            "transmitter_frequency": 410.085,
            "start_center_frequency": 410.0864453125,
            "channel_bandwidth": 0.08544921875,
            "samples_per_second": 85449.21875,
            "name": "SURVEY_53240",
            "duration": 4590.0,
            "id": ObjectId("62e135df156c338ed328aadd"),
            "notifications": True,
            "type": "observation",
            "start_time": "2022-07-28 05:20:29+00:00",
            "target_name": 53240
        },
        "detection": {
            "save_candidates": True,
            "debug_candidates": False,
            "beam_range": [
                0,
                32
            ],
            "multi_proc": True,
            "n_procs": 12,
            "select_highest_snr": True,
            "n_noise_samples": 5,
            "noise_channels": [
                1000,
                4000
            ],
            "noise_use_rms": False,
            "enable_doppler_window": True,
            "doppler_range": [
                -19688,
                19507
            ],
            "enable_gradient_thold": True,
            "gradient_thold": [
                -0.2,
                -2901.47
            ],
            "save_tdm": True,
            "similarity_thold": 0.1,
            "linearity_thold": 0.95,
            "filter_transmitter": False
        },
        "loggers": {
            "keys": "root"
        },
        "handlers": {
            "keys": "stream_handler"
        },
        "formatters": {
            "keys": "formatter"
        },
        "handler_stream_handler": {
            "class": "StreamHandler",
            "formatter": "formatter",
            "args": "(sys.stderr,)"
        },
        "handler_rot_handler": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "formatter",
            "args": "()"
        },
        "formatter_formatter": {
            "format": "%(asctime)s %(levelname)-8s %(process)-5s %(processName)-15s %(threadName)-20s %(message)s"
        },
        "birales": {
            "debug_level": "DEBUG"
        },
        "monitoring": {
            "visualize_beams": [],
            "file_path": "public",
            "save_filtered_beam_data": False,
            "image_ext": "png"
        },
        "beamplotter": {
            "beam_range": 15,
            "nof_samples": 2048
        },
        "flask": {
            "debug": True,
            "secret_key": "secret!",
            "host": "0.0.0.0",
            "port": 8000
        },
        "target": {
            "name": 27944
        }
    },
    "log_filepath": "/var/log/birales/2022_07_28/SURVEY_53240.log",
    "noise_mean": 0.984532326459885,
    "sampling_time": 0.0958698057142857,
    "tx": 410.085
}
obs_info_file = "/storage/data/birales/2022_07_28/SURVEY_53240/SURVEY_53240_raw.dat.pkl"
obs_info_file_old = "/storage/data/birales/2022_07_28/SURVEY_2/SURVEY_2_raw.dat.pkl"
# Write observation information
# with open(obs_info_file, 'wb') as f:
#     pickle.dump(obs_info_str, f)

with open(obs_info_file_old, 'rb') as f:
    obs_info = pickle.load(f)
    print(obs_info['timestamp'])



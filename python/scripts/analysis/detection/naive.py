# -*- coding: utf-8 -*-

"""
Naive detection algorithm.

This script implemented hte Naive detection algorithm used in the PyBirales detection module.
The algorithm uses a series of filters before it runs the clustering algorithm to
extract the RSO tracks.

Use the Detection Analysis.ipynb notebook for analysing the results

"""
from astropy.io import fits
import os
from pybirales.pipeline.modules.detection.filter import RemoveBackgroundNoiseFilter, RemoveTransmitterChannelFilter, PepperNoiseFilter
from pybirales.pipeline.modules.detection.detector import Detector

HOME_DIR = os.environ['HOME']
FITS_DIR = os.path.join(HOME_DIR, '.birales/visualisation/fits/')
FITS_FILENAME = 'Observation_2018-04-11T1010/Observation_2018-04-11T1010_filtered.fits'

candidates = []
_iter_count = 0

# Read the fits raw data
input_data = fits.getdata(FITS_FILENAME)
obs_info = {}

# Filter the data (Background noise and Pepper noise filter)
filters = [
    RemoveBackgroundNoiseFilter(std_threshold=4.),
    RemoveTransmitterChannelFilter(),
    PepperNoiseFilter(),
]

for f in filters:
    f.apply(input_data, obs_info)


# DBSCAN clustering algorithm
# Pre-process the input data
input_data, channels, channel_noise = _pre_process(input_data, obs_info)

# Process the input data and identify the detection clusters
clusters = _get_clusters(input_data, channels, channel_noise, obs_info, _iter_count)



# Create new tracks from clusters or merge clusters into existing tracks
candidates = _aggregate_clusters(candidates, clusters, obs_info)

# Check each track and determine if the detection object has transitted outside FoV
candidates = _active_tracks(candidates, _iter_count)


# Extract tracks and compare with the expected

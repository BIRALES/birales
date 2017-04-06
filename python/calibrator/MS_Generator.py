import numpy as np
import os
from datetime import datetime
from Calibrator import Builder
from Calibrator import Parser
from Calibrator import RunOSKAR
from Calibrator import MSReader
from Calibrator import RedBasFinder



### Config Setup Creator

## Initialize Setup Builder
build = Builder()

user_defined_name = 'BEST_2'
TM_name = 'I_O/' + user_defined_name + '.tm'

date_save = datetime.now()

## Sky Model Settings
# SM Path
build.sky.Dir = 'I_O/SkyModel/SkyModel.osm'

## Observation Settings
# Phase Centre (Pointing) RA, Dec
build.observation.PCRA = 30
build.observation.PCDec = 90
# Observation Starts at "TStart" for an Observation of time "Length" divided into "TSteps"
build.observation.TSteps = 1
build.observation.TStart = str(date_save.day) + '-' + str(date_save.month) + '-' + str(date_save.year) + ' ' + str(date_save.hour) + ':' + str(date_save.minute) + ':' + str(date_save.second) + '.000'
build.observation.Length = '00:01:00.000'
# No of channels, starting from "StartFreq" (in mHz) with increments "IncreFreq" (in mHz) per channel
build.observation.Chan = 1
build.observation.StartFreq = 400
build.observation.IncreFreq = 1

## Telescope Settings
# TM Directory
build.telescope.Dir = TM_name
# Telescope Longitude, Latitude
build.telescope.Long = 0
build.telescope.Lati = 90
# Telescope Polarization Mode
build.telescope.PolM = 'Scalar'

## Interferometer Settings
# Visibility File Path
build.interferometer.VisF_Dir = 'I_O/' + user_defined_name + '.vis'
# Bandwith (in mHz) and Time Averaging (in sec)
build.interferometer.Bandwith = 1
build.interferometer.TAverage = 1
build.interferometer.MS_Dir= 'I_O/' + user_defined_name + '.ms'

## Beam Pattern Settings
# BP Image Path
build.beam_pattern.Dir = 'I_O/' + user_defined_name
# BP Image Square FOV (deg) and Square Size (Pixels)
build.beam_pattern.Img_FOV = 180
build.beam_pattern.Img_Size = 512
# BP Reference Coordinate Frame
build.beam_pattern.CoordFrame = 'Horizon'

## Sky Image Settings
# Sky Image Path
build.image.Dir = 'I_O/' + user_defined_name
# Sky Image Square FOV (deg) and Square Size (Pixels)
build.image.Img_FOV = 30
build.image.Img_Size = 512
# Sky Image Type
build.image.Img_Type = 'I'

## Initialize Setup Parser
parse = Parser()
# Run Parser, Save Setup to OSKAR .ini file
setup_filepath = 'I_O/' + 'setup.ini'
parse.parse_to_ini(setup_filepath)



### Generate MS with UVW

## Initialize OSKAR Runner
OSK = RunOSKAR(setup_filepath)

## OSKAR Interferometer Run for Generating MS with UVW and Visibilities
OSK.interferometer_run()



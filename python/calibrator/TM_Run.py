import numpy as np
from Calibrator import TM_Generator


### Telescope Model Generator

## Initialize TM Generator
TM = TM_Generator()

#antennas = [[-34.8875321, 6.804708, 0],
#                    [-34.8875321, 34.023538, 0],
#                    [-34.8875321, 61.242368, 0],
#                    [-34.8875321, 88.461198, 0],
#                    [-34.8875321, 20.414123, 0],
#                    [-34.8875321, 47.632953, 0],
#                    [-34.8875321, 74.851783, 0],
#                    [-34.8875321, 102.070613, 0],
#                    [-26.8875321, 6.804708, 0],
#                    [-26.8875321, 34.023538, 0],
#                    [-26.8875321, 61.242368, 0],
#                    [-26.8875321, 88.461198, 0],
#                    [-26.8875321, 20.414123, 0],
#                    [-26.8875321, 47.632953, 0],
#                    [-26.8875321, 74.851783, 0],
#                    [-26.8875321, 102.070613, 0],
#                    [-18.8875321, 6.804708, 0],
#                    [-18.8875321, 34.023538, 0],
#                    [-18.8875321, 61.242368, 0],
#                    [-18.8875321, 88.461198, 0],
#                    [-18.8875321, 20.414123, 0],
#                    [-18.8875321, 47.632953, 0],
#                    [-18.8875321, 74.851783, 0],
#                    [-18.8875321, 102.070613, 0],
#                    [-10.8875321, 6.804708, 0],
#                    [-10.8875321, 34.023538, 0],
#                    [-10.8875321, 61.242368, 0],
#                    [-10.8875321, 88.461198, 0],
#                    [-10.8875321, 20.414123, 0],
#                    [-10.8875321, 47.632953, 0],
#                    [-10.8875321, 74.851783, 0],
#                    [-10.8875321, 102.070613, 0]]

#antennas = [[-34.8875321, 102.070613, 0],
#                    [-34.8875321, 20.414123, 0],
#                    [-34.8875321, 47.632953, 0],
#                    [-34.8875321, 74.851783, 0],
#                    [-34.8875321, 6.804708, 0],
#                    [-34.8875321, 34.023538, 0],
#                    [-34.8875321, 61.242368, 0],
#                    [-34.8875321, 88.461198, 0],
#                    [-26.8875321, 102.070613, 0],
#                    [-26.8875321, 20.414123, 0],
#                    [-26.8875321, 47.632953, 0],
#                    [-26.8875321, 74.851783, 0],
#                    [-26.8875321, 6.804708, 0],
#                    [-26.8875321, 34.023538, 0],
#                    [-26.8875321, 61.242368, 0],
#                    [-26.8875321, 88.461198, 0],
#                    [-18.8875321, 102.070613, 0],
#                    [-18.8875321, 20.414123, 0],
#                    [-18.8875321, 47.632953, 0],
#                    [-18.8875321, 74.851783, 0],
#                    [-18.8875321, 6.804708, 0],
#                    [-18.8875321, 34.023538, 0],
#                    [-18.8875321, 61.242368, 0],
#                    [-18.8875321, 88.461198, 0],
#                    [-10.8875321, 102.070613, 0],
#                    [-10.8875321, 20.414123, 0],
#                    [-10.8875321, 47.632953, 0],
#                    [-10.8875321, 74.851783, 0],
#                    [-10.8875321, 6.804708, 0],
#                    [-10.8875321, 34.023538, 0],
#                    [-10.8875321, 61.242368, 0],
#                    [-10.8875321, 88.461198, 0]]

antennas = [[-8.499750, -35.000000, 0.000000],
                    [-2.833250, -35.000000, 0.000000],
                    [2.833250, -35.000000, 0.000000],
                    [8.499750, -35.000000, 0.000000],
                    [-8.499750, -25.000000, 0.000000],
                    [-2.833250, -25.000000, 0.000000],
                    [2.833250, -25.000000, 0.000000],
                    [8.499750, -25.000000, 0.000000],
                    [-8.499750, -15.000000, 0.000000],
                    [-2.833250, -15.000000, 0.000000],
                    [2.833250, -15.000000, 0.000000],
                    [8.499750, -15.000000, 0.000000],
                    [-8.499750, -5.000000, 0.000000],
                    [-2.833250, -5.000000, 0.000000],
                    [2.833250, -5.000000, 0.000000],
                    [8.499750, -5.000000, 0.000000],
                    [-8.499750, 5.000000, 0.000000],
                    [-2.833250, 5.000000, 0.000000],
                    [2.833250, 5.000000, 0.000000],
                    [8.499750, 5.000000, 0.000000],
                    [-8.499750, 15.000000, 0.000000],
                    [-2.833250, 15.000000, 0.000000],
                    [2.833250, 15.000000, 0.000000],
                    [8.499750, 15.000000, 0.000000],
                    [-8.499750, 25.000000, 0.000000],
                    [-2.833250, 25.000000, 0.000000],
                    [2.833250, 25.000000, 0.000000],
                    [8.499750, 25.000000, 0.000000],
                    [-8.499750, 35.000000, 0.000000],
                    [-2.833250, 35.000000, 0.000000],
                    [2.833250, 35.000000, 0.000000],
                    [8.499750, 35.000000, 0.000000]]

## Parse Antenna Positions
TM.antenna_parser(antennas)

## Create TM
user_defined_name = 'BEST_2'
name = 'I_O/' + user_defined_name + '.tm'
TM.create_TM(name)

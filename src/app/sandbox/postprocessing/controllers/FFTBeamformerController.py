# -*- coding: utf-8 -*-

import xlrd
import numpy as np

n_x = 8  # Number of cylinders
d_x = 13.6  # Distance of the cylinders
n_y = 4  # Number of receivers in 1 cylinder
d_y = 8  # Distance of receivers in 1 cylinder

n_dipoles = 16  # ?
d_dipoles = 0.5  # ?
offset = 1e-6  # Offset in radians vs. limit angle of the dipole

min_db = -80.

h_plane_ns_cyl = np.genfromtxt('../data/NS_PianoH.csv', delimiter = ',')
angles = h_plane_ns_cyl[:, 1]  # Get second column in the csv (angles)

pattern_db = h_plane_ns_cyl[:, 2]  # Get third column in the csv (db)

h_plane_pattern_cyl = np.power(10., (pattern_db * 0.05))

h_plane_pattern_cyl /= np.max(h_plane_pattern_cyl)  # normalisation of the data

# Initialisation of spatial variables for 3D XYZ plot
theta_x_min = -4
theta_x_max = +4
theta_x_step = 1
MM = 81

theta_y_min = -4
theta_y_max = +4
theta_y_step = 1
NN = 81

dtor = np.pi / 180.  # conversion factor in radians

theta_x_min_rad = theta_x_min * dtor
theta_x_max_rad = theta_x_max * dtor
theta_x_step_rad = theta_x_step / 180 * np.pi

theta_y_min_rad = theta_y_min * dtor
theta_y_max_rad = theta_y_max * dtor
theta_y_step_rad = theta_y_step / 180 * np.pi

# Construction of the spatial plane for plotting in 3D space
theta_x = np.linspace(theta_x_min, theta_x_max, MM)
theta_y = np.linspace(theta_y_min, theta_y_max, MM)

theta_x_rad = np.linspace(theta_x_min_rad, theta_x_max_rad, MM)
theta_y_rad = np.linspace(theta_y_min_rad, theta_y_max_rad, MM)

theta_X, theta_Y = np.meshgrid(theta_x, theta_y)
theta_X_rad, theta_Y_rad = np.meshgrid(theta_x_rad, theta_y_rad)

# Array geometry matrix (placement of antennas)
x = np.zeros((n_x, n_y))
y = np.zeros((n_x, n_y))

for m in range(0, n_y):
    for n in range(0, n_x):
        x[n, m] = (n - 1) * d_x

for n in range(0, n_x):
    for m in range(0, n_y):
        y[n, m] = (m - 1) * d_y

n = n_x * n_y  # Number of antennas that make up the matrix

xy = np.zeros((n, 2))
xy[:, 0] = np.reshape(x, n)
xy[:, 1] = np.reshape(y, n)

# Beam pattern calculation for the dipole
beam_pattern_dipole = np.cos((np.pi / 2) * np.sin(theta_y_rad + offset))
beam_pattern_dipole = beam_pattern_dipole / np.cos(theta_y + offset)
beam_pattern_dipole = np.abs(beam_pattern_dipole) ** 2.

# Calculation for the group factor of the dipoles
dip = np.zeros((n_dipoles, len(theta_y_rad)))

for a in range(0, n_dipoles):
    for b in range(0, len(theta_y_rad)):
        dip[a, b] = np.exp(2j * np.pi * (a - 1) * d_dipoles * np.sin(theta_y_rad[b]))

# L'ACCOPPIAMENTO TRA I DIPOLI E' FISSO (PUNTAMENTO = 0°)
w_dipole = np.ones((n_dipoles, 1)) / n_dipoles
group_dipole_factor = np.abs(w_dipole.conj().T * dip) ** 2.

beam_pattern_sensor = beam_pattern_dipole * group_dipole_factor

# Calculation of the signal beam of BEST-2 element
beam_element = h_plane_ns_cyl * beam_pattern_sensor
beam_element = 10 * np.log(beam_element)

beam_pattern_sensor = beam_pattern_sensor.getT()

af3d = np.zeros(n, len(theta_x_rad), len(theta_y_rad))
for b in range(0, len(theta_y_rad)):
    for a in range(0, len(theta_x_rad)):
        for _n in range(0, n):
            h = np.sin(theta_x_rad[a])
            g = xy[_n, 0] * h
            k = np.sin(theta_y_rad[b])

            l = g + (xy[_n, 2] * k)

            af3d[_n, a, b] = np.exp(2j * np.pi * l)

af3d_mod = np.reshape(af3d, n_x, n_y, len(theta_x_rad), len(theta_y_rad))

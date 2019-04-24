# -*- coding: utf-8 -*-

import os

import pandas as pd

# Input arguments
save_dir = "/home/denis/Desktop/"
filepath = "/home/denis/Desktop/far_field_array_410MHz_0.1deg.txt"
frequency = "array_410"

export_filepath_mask = os.path.join(save_dir, "far_field_{}MHz_dt={}_dp={}.csv")

col_names = [
    "Theta", "Phi", "Gain", "Abs(Theta)", "Phase(Theta)", "Abs(Phi)", "Phase(Phi)", "Ax.Ratio"
]
df = pd.read_csv(filepath, sep="\s+", skiprows=2, header=None, names=col_names)

n_deg_theta = 180.
n_deg_phi = 360.
n_rows = len(df.index)

res_theta = df.diff(axis=0)['Theta'][1]
res_phi = ((n_deg_theta / res_theta + 1) * n_deg_phi) / n_rows

# Sort the data by theta
df = df.sort_values(by=['Theta', 'Phi'])

export_filepath = export_filepath_mask.format(frequency, res_theta, res_phi)

# Export it in a format that Matlab expects
df[["Theta", "Phi", "Gain"]].to_csv(path_or_buf=export_filepath, header=False, index=False)

print('Imported CST file from {}.\nExported {} data points exported with Δθ={:2.1f}° and ΔФ={:2.1f}°. '
      '\nExported at: {}.'.format(
    os.path.basename(filepath),
    n_rows,
    res_theta,
    res_phi,
    export_filepath
))

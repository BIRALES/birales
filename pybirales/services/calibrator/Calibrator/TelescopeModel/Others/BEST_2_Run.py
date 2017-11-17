import os

interf_run = "/usr/local/bin/./oskar_sim_interferometer setup.ini"
os.system(interf_run)

#beampt_run = "/usr/local/bin/./oskar_sim_beam_pattern setup.ini"
#os.system(beampt_run)

# msrset_run = "/usr/local/bin/./oskar_vis_to_ms BEST_2.vis"
# os.system(msrset_run)

#imager_run = "/usr/local/bin/./oskar_imager setup.ini"
#os.system(imager_run)

CASA_run = "/opt/CASA/bin/./casa -c BEST_2_MS_to_H5.py" 
os.system(CASA_run)




import os
import numpy as np


class RunOSKAR:

    def __init__(self, setup):

        self.setup = setup

    def interferometer_run(self):

        os_command = "oskar_sim_interferometer " + self.setup
        os.system(os_command)

    def beam_pattern_run(self):

        # For Debugging
        os_command = "oskar_sim_beam_pattern " + self.setup
        os.system(os_command)

    def imager_run(self):

        # For Debugging
        os_command = "oskar_imager " + self.setup
        os.system(os_command)        
        

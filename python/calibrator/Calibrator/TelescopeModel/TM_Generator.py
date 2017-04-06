import os
import numpy as np


class TM_Generator:

    def __init__(self):
        
        self.grid_pos_x = None
        self.grid_pos_y = None
        self.grid_pos_z = None

    def antenna_parser(self, ants):

        # Best-2 Grid Antennas as defined, in order
    
        antennas = np.array(ants)

        self.grid_pos_x = antennas[:, 0]
        self.grid_pos_y = antennas[:, 1]
        self.grid_pos_z = antennas[:, 2]

    def create_TM(self, name):

        TM = name
        os.mkdir(TM)

        stat_l = TM + "/layout.txt"
        layout_file = open(stat_l, "a")
        for i in range(len(self.grid_pos_x)):
            layout_file.write('%f' % self.grid_pos_x[i] + ', ' + '%f' % self.grid_pos_y[i] + ', ' + '%f' % self.grid_pos_z[i] + '\n')
        layout_file.close()

        tripleDig = [format(x, "03d") for x in range(len(self.grid_pos_x))]
        for i in range(len(self.grid_pos_x)):
	
	    SF = TM + "/station" + tripleDig[i]
	    os.mkdir(SF)
	
	    LF = SF + "/layout.txt"
	    text_file = open(LF, "a")
	    text_file.write('0.0, 0.0, 0.0')
	    text_file.close()


import math
import os
import numpy as np

#Best-2_Grid
grid_pos_x= [-34.8875321, -26.8875321, -18.8875321, -10.8875321, -34.8875321, -26.8875321, -18.8875321, -10.8875321, -34.8875321, -26.8875321, -18.8875321, -10.8875321, -34.8875321, -26.8875321, -18.8875321, -10.8875321, -34.8875321, -26.8875321, -18.8875321, -10.8875321, -34.8875321, -26.8875321, -18.8875321, -10.8875321, -34.8875321, -26.8875321, -18.8875321, -10.8875321, -34.8875321, -26.8875321, -18.8875321, -10.8875321]
grid_pos_y= [6.804708, 6.804708, 6.804708, 6.804708, 20.414123, 20.414123, 20.414123, 20.414123, 34.023538, 34.023538, 34.023538, 34.023538, 47.632953, 47.632953, 47.632953, 47.632953, 61.242368, 61.242368, 61.242368, 61.242368, 74.851783, 74.851783, 74.851783, 74.851783, 88.461198, 88.461198, 88.461198, 88.461198, 102.070613, 102.070613, 102.070613, 102.070613]
grid_pos_z=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 

# Actual_Config_Centre 
centre_x = ((sum(grid_pos_x))/(len(grid_pos_x)))
centre_y = ((sum(grid_pos_y))/(len(grid_pos_y)))

dis = list()

for k in range(0, len(grid_pos_x)):
    lon1 = centre_x
    lat1 = centre_y
    lon2 = (grid_pos_x[k])
    lat2 = (grid_pos_y[k])
    point1 = lon1
    point2 = lat1
    point3 = lon2
    point4 = lat2
    dis.append(math.sqrt(((lon2-lon1) ** 2)+((lat2-lat1) ** 2)))

centre = dis.index((min(dis)))

# Create_x_y_Distance_Format
xDistance = list()
yDistance = list()
for i in range(0, len(grid_pos_x)):
    lon1 = (grid_pos_x[centre])
    lat1 = (grid_pos_y[centre])
    lon2 = (grid_pos_x[i])
    lat2 = (grid_pos_y[i])
    xDistance.append(lat2 - lat1)
    yDistance.append(lon2 - lon1)

xyVals = list()
for i in range(0, len(grid_pos_x)):
    xyVals.append((grid_pos_x[i], grid_pos_y[i], grid_pos_z[i]))

# Create_TM
TM = "BEST_2.tm"
os.mkdir(TM)

stat_l = TM + "/layout.txt"
layout_file = open(stat_l, "a")
for i in range(len(xyVals)):
    layout_file.write('%f' % xyVals[i][0] + ', ' + '%f' % xyVals[i][1] + ', ' + '%f' % xyVals[i][2] + '\n')
layout_file.close()

tripleDig = [format(x, "03d") for x in range(len(xyVals))]

for i in range(len(xyVals)):
	
	SF = TM + "/station" + tripleDig[i]
	os.mkdir(SF)
	
	LF = SF + "/layout.txt"
	text_file = open(LF, "a")
	text_file.write('0.0, 0.0, 0.0')
	text_file.close()
	
	GPE = SF + "/gainphase.txt"
	randGain = float(np.random.uniform(1, 1.5, 1))
	randPhase = float(np.random.uniform(1, 1.5, 1))
	stdG = float(np.random.uniform(1, 1.5, 1))
	stdP = float(np.random.uniform(1, 1.5, 1))
	gainphase_file = open(GPE, "a")
	gainphase_file.write('%f' % randGain + ', ' + '%f' % randPhase + ', ' + '%f' % stdG + ', ' + '%f' % stdP)
	gainphase_file.close()





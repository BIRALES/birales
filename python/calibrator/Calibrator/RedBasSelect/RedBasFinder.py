import numpy as np
import scipy.spatial


class RedBasFinder:

    def __init__(self, UVW, antenna1, antenna2):

        self.UVW = UVW
        self.Bas_1 = None
        self.Bas_2 = None
        self.basl_ind = []
        self.basl_grp = []
        self.ant1 = antenna1
        self.ant2 = antenna2

    def find(self):

        # Define redundant baselines and obtain pairs of redundant baselines, sorted
        redbas = scipy.spatial.cKDTree(self.UVW).query_pairs(r=0.1, p=np.infty)
        redbas = np.array(list(redbas))
        index_1 = redbas[:, 0]
        index_2 = redbas[:, 1]
        indices_out = sorted(zip(index_1, index_2), key=lambda x: x[0])
        indices_out = np.array(indices_out)

        self.Bas_1 = indices_out[:,0]
        self.Bas_2 = indices_out[:,1]

        # Group all inter-redundant baselines by same group number
        grp_num = -1
        for i in range(0, len(self.Bas_1)):
	    if ((self.Bas_1[i] not in self.basl_ind) and (self.Bas_2[i] not in self.basl_ind)):
                grp_num += 1
		self.basl_ind.append(self.Bas_1[i])
		self.basl_ind.append(self.Bas_2[i])
		self.basl_grp.append(grp_num)
		self.basl_grp.append(grp_num)
	    if ((self.Bas_1[i] in self.basl_ind) and (self.Bas_2[i] not in self.basl_ind)):
		self.basl_ind.append(self.Bas_2[i])
		self.basl_grp.append(grp_num)
	    if ((self.Bas_1[i] not in self.basl_ind) and (self.Bas_2[i] in self.basl_ind)):
                grp_num += 1
		self.basl_ind.append(self.Bas_1[i])
		self.basl_grp.append(grp_num)

        #Add_other_non_redundant_unique_baselines_and_their_indices
        for i in range(0, len(self.basl_ind)):
	    if (i not in self.basl_ind):
		grp_num += 1
		self.basl_ind.append(i)
		self.basl_grp.append(grp_num)

        #Sort_baselines_by_baseline_number_for_A_matrix
        basl_sort = sorted(zip(self.basl_ind, self.basl_grp), key=lambda x: x[0])
        basl_sort = np.array(basl_sort)
        self.basl_ind = basl_sort[:,0]
        self.basl_grp = basl_sort[:,1] 

    def parse(self, name):

        text_file = open(name, "w")
        for i in range(len(self.basl_ind)):   
            text_file.write(str(self.basl_ind[i]) + ' ' + str(self.ant1[self.basl_ind[i]]) + ' ' + str(self.ant2[self.basl_ind[i]]) + ' ' + str(self.basl_grp[i]) + '\n')
        text_file.close()  


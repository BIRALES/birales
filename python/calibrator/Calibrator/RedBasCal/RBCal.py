import numpy as np
import math as ma


class RBCal:

    def __init__(self):
         
        self.data = None
        self.reals = []
        self.imags = []
        self.RB_index = None
        self.RB_ants1 = None
        self.RB_ants2 = None
        self.RB_group = None
        self.real_data = []
        self.imag_data = []
        self.gain_coeff = None
        self.phase_coeff = None

    def load_RBs(self, filename):

        RBs = np.loadtxt(filename)
        self.RB_index = RBs[:, 0]
        self.RB_ants1 = RBs[:, 1]
        self.RB_ants2 = RBs[:, 2]
        self.RB_group = RBs[:, 3]

    def calib(self, data):

        self.data = data
        
        # Forming A Configuration Matrix
        ant_no = max(self.RB_ants2) + 1
        bas_no = max(self.RB_index) + 1

        A = np.zeros((len(self.RB_index),(ant_no + bas_no)))

        for i in range(len(self.RB_index)):
            A[i,(self.RB_ants1[i])] = 1
            A[i,(self.RB_ants2[i])] = 1
            A[i,(ant_no + self.RB_index[i])] = 1

        # Forming Real and Imag Data

        self.reals = []
        self.imags = []
        self.real_data = []
        self.imag_data = []

        for i in range(len(self.RB_index)):
            reals = self.data[0,self.RB_index[i],0].real
            imags = self.data[0,self.RB_index[i],0].imag

            self.reals.append(reals)
            self.imags.append(imags)
            self.real_data.append(np.log((ma.pow((reals), 2)) + (ma.pow((imags), 2))))
            self.imag_data.append(ma.atan(imags/reals))

        self.real_data = np.array(self.real_data)
        self.imag_data = np.array(self.imag_data)

        #self.reals = np.array(self.reals)
        #self.imags = np.array(self.imags)

        # Forming N Cov Matrix
        N_R = np.zeros((len(self.RB_index), len(self.RB_index)))
	N_I = np.zeros((len(self.RB_index), len(self.RB_index)))
	for i in range(len(self.RB_index)):
            N_R[i,i] = 1
            N_I[i,i] = 1

        # Solving for x_coeff with Least Squares Estimation
        A_trans = [list(i) for i in zip(*A)]
        A_trans = np.array(A_trans)
	N_inv_R = np.linalg.inv(N_R)
	N_inv_I = np.linalg.inv(N_I)

        gain_coeff = (np.dot((np.linalg.inv(np.dot((np.dot(A_trans,N_inv_R)),A))),(np.dot((np.dot(A_trans,N_inv_R)),self.real_data))))
	phase_coeff = (np.dot((np.linalg.inv(np.dot((np.dot(A_trans,N_inv_I)),A))),(np.dot((np.dot(A_trans,N_inv_I)),self.imag_data))))

        #gain_coeff=(np.dot((np.dot((np.dot((np.linalg.inv(np.dot((np.dot(A_trans,N_inv_R)),A))),A_trans)),N_inv_R)),self.real_data))
	#phase_coeff=(np.dot((np.dot((np.dot((np.linalg.inv(np.dot((np.dot(A_trans,N_inv_I)),A))),A_trans)),N_inv_I)),self.imag_data))
        
        gains = gain_coeff[0:ant_no]
        minimum_pv = 0
        maximum_pv = max(gains)
        minimum_nv = min(gains)
        maximum_nv = 0
        for i in range(len(gains)):
            if gains[i] >= 0:
                gains[i] = (gains[i] - minimum_pv) / (maximum_pv - minimum_pv)
            if gains[i] < 0:
                gains[i] = ((gains[i] - minimum_nv) / (maximum_nv - minimum_nv)) - 1     
        self.gain_coeff = gains

        self.phase_coeff = phase_coeff[0:ant_no]         
        
    def parse_coeff(self, filename1, filename2):

        text_file = open(filename1, "w")
        for i in range(len(self.gain_coeff)):   
            text_file.write(str(i) + ' ' + str(self.gain_coeff[i]) + '\n')
        text_file.close()

        text_file = open(filename2, "w")
        for i in range(len(self.phase_coeff)):   
            text_file.write(str(i) + ' ' + str(self.phase_coeff[i]) + '\n')
        text_file.close()          
        

        
        
        
        

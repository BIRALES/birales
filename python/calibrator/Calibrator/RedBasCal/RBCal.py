import numpy as np
import math as ma
import h5py
import matplotlib.pyplot as plt


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
        self.gain_coeff_all = None
        self.phase_coeff_all = None
        self.gain_coeff_final = []
        self.phase_coeff_final = []

    def load_RBs(self, filename):

        RBs = np.loadtxt(filename, dtype=np.int)
        self.RB_index = RBs[:, 0]
        self.RB_ants1 = RBs[:, 1]
        self.RB_ants2 = RBs[:, 2]
        self.RB_group = RBs[:, 3]

    def calib(self, filename):

        np.set_printoptions(precision=5)

        # Open Correlated Visibilities File
        f = h5py.File(filename, "r")
        data = f["Vis"]

        # Define size of data matrix to hold visibilities for all baselines at a given time sample
        data_out = np.zeros((1, len(self.RB_index), 1), dtype=np.complex)

        # Select number of time samples to carry calibration over (1 = approx 1.5s)
        time = 200

        # Select start time sample for calibration, if required
        time_peak = np.where(data[:,0,:,0].real == np.max(data[:,0,:,0].real))[0][0]
        start = int(time_peak-(time))
        if start < 0:
            start = 0

        start = 600

        self.gain_coeff_all = np.zeros((time, np.max(self.RB_ants2)+1))
        self.phase_coeff_all = np.zeros((time, np.max(self.RB_ants2)+1))  
        
        for t in range(0, time):
            # Per-Baseline Integration in Time
            for i in range(len(self.RB_index)):
                data_out[0, i, 0] = ((data[t+start, 0, i, 0]))#/(len(data[:,0,i,0]))
            self.data = data_out

            # Forming A Configuration Matrix
            ant_no = max(self.RB_ants2) + 1
            bas_no = max(self.RB_group) + 1

            A = np.zeros((len(self.RB_index), int(ant_no + bas_no)))
    
            for i in range(len(self.RB_index)):
                A[i, (self.RB_ants1[i])] = 1
                A[i, (self.RB_ants2[i])] = 1
                A[i, (ant_no + self.RB_group[i])] = 1

            # Forming Real and Imag Data

            self.reals = []
            self.imags = []
            self.real_data = []
            self.imag_data = []

            for i in range(len(self.RB_index)):
                reals = self.data[0, int(self.RB_index[i]), 0].real
                imags = self.data[0, int(self.RB_index[i]), 0].imag

                self.reals.append(reals)
                self.imags.append(imags)
                self.real_data.append(np.log(np.sqrt((ma.pow((np.float(reals)), 2)) + (ma.pow((np.float(imags)), 2)))))
                self.imag_data.append((np.arctan(np.float(imags) / np.float(reals))))

            self.real_data = np.array(self.real_data)
            self.imag_data = np.array(self.imag_data)

            self.reals = np.array(self.reals)
            self.imags = np.array(self.imags)

            # Forming N Cov Matrix

            N_R = np.zeros((len(self.RB_index), len(self.RB_index)))
            N_I = np.zeros((len(self.RB_index), len(self.RB_index)))
            for i in range(len(self.RB_index)):
                N_R[i,i] = 1/((np.sqrt((ma.pow((np.float(self.reals[i])), 2)) + (ma.pow((np.float(self.imags[i])), 2))))**2)
                N_I[i,i] = -1/((np.sqrt((ma.pow((np.float(self.reals[i])), 2)) + (ma.pow((np.float(self.imags[i])), 2))))**2)

            # Solving for x_coeff with Least Squares Estimation
            A_trans = [list(i) for i in zip(*A)]
            A_trans = np.array(A_trans)
            N_inv_R = np.linalg.pinv(N_R)
            N_inv_I = np.linalg.pinv(N_I)

            gain_coeff = (np.dot((np.linalg.pinv(np.dot((np.dot(A_trans, N_inv_R)), A))),
                             (np.dot((np.dot(A_trans, N_inv_R)), self.real_data))))
            phase_coeff = np.degrees(np.dot((np.linalg.pinv(np.dot((np.dot(A_trans, N_inv_I)), A))),
                               (np.dot((np.dot(A_trans, N_inv_I)), self.imag_data))))

            gains = gain_coeff[0:ant_no]
            gains = gains / gains[4]
            self.gain_coeff = gains

            phases = phase_coeff[0:ant_no]
            phases = phases - phases[4]
            self.phase_coeff = phases
            
            # Maintain all coefficients for averaging
            for i in range(len(self.gain_coeff)):
                self.gain_coeff_all[t,i] = self.gain_coeff[i]
                self.phase_coeff_all[t,i] = self.phase_coeff[i]

            print(t)

        # Obtain coeffs
        for i in range(len(self.gain_coeff)):
            self.gain_coeff_final.append(np.mean(self.gain_coeff_all[:,i]))
            self.phase_coeff_final.append(np.mean(self.phase_coeff_all[:,i]))

        print("Gain_coeff " + str(self.gain_coeff_final))
        print("Phase_coeff " + str(self.phase_coeff_final))

    def parse_coeff(self, filename1, filename2):

        text_file = open(filename1, "w")
        for i in range(len(self.gain_coeff_final)):
            text_file.write('a' + str(i) + ' ' + str(self.gain_coeff_final[i]) + '\n')
        text_file.close()

        text_file = open(filename2, "w")
        for i in range(len(self.phase_coeff_final)):
             text_file.write('a' + str(i) + ' ' + str(self.phase_coeff_final[i]) + '\n')
        text_file.close()

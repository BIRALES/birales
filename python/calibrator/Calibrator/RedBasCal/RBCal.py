import numpy as np
import math as ma
import h5py


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

        RBs = np.loadtxt(filename, dtype=np.int)
        self.RB_index = RBs[:, 0]
        self.RB_ants1 = RBs[:, 1]
        self.RB_ants2 = RBs[:, 2]
        self.RB_group = RBs[:, 3]

    def calib(self, filename):

        np.set_printoptions(precision=5)

        f = h5py.File(filename, "r")
        data = f["Vis"]
        data_out = np.zeros((1, len(self.RB_index), 1), dtype=np.complex)

        # Per-Baseline Integration in Time
        for i in range(len(self.RB_index)):
            data_out[0, i, 0] = (sum(data[:, 0, i, 0]))#/(len(data[:,0,i,0]))
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
            self.imag_data.append(np.degrees(np.arctan(np.float(imags) / np.float(reals))))

        self.real_data = np.array(self.real_data)
        self.imag_data = np.array(self.imag_data)

        self.reals = np.array(self.reals)
        self.imags = np.array(self.imags)

        # Forming N Cov Matrix

        N_R = np.zeros((len(self.RB_index), len(self.RB_index)))
        N_I = np.zeros((len(self.RB_index), len(self.RB_index)))
        #for i in range(len(self.RB_index)):
        #    self.RB_group = np.array(self.RB_group)
        #    indices = np.where(self.RB_group == self.RB_group[int(self.RB_index[i])])[0]
        #    redbasl_data_r = []
        #    redbasl_data_i = []
        #    indices = np.array(indices)
        #    for j in indices:
        #        reald = self.real_data[int(self.RB_index[j])]
        #        redbasl_data_r.append(reald)
        #        imagd = self.imag_data[int(self.RB_index[j])]
        #        redbasl_data_i.append(imagd)
        #    redbasl_mean_r = np.float(np.mean(redbasl_data_r))
        #    redbasl_mean_i = np.float(np.mean(redbasl_data_i))
        #    redbasl_diff_r = []
        #    redbasl_diff_i = []
        #    for k in indices:
        #        redbasl_diff_r.append((self.real_data[int(self.RB_index[k])] - redbasl_mean_r) ** 2)
        #        redbasl_diff_i.append((self.imag_data[int(self.RB_index[k])] - redbasl_mean_i) ** 2)
        #    redbasl_diff_total_r = np.sum(redbasl_diff_r)
        #    redbasl_diff_total_i = np.sum(redbasl_diff_i)
        #    spread_r = (1 / np.float(len(indices))) * redbasl_diff_total_r
        #    spread_i = (1 / np.float(len(indices))) * redbasl_diff_total_i
        #    N_R[i, i] = spread_r / (self.real_data[int(self.RB_index[i])] ** 2)
        #    N_I[i, i] = spread_i / (self.imag_data[int(self.RB_index[i])] ** 2)

        for i in range(len(self.RB_index)):
            N_R[i,i] = 1/((np.sqrt((ma.pow((np.float(self.reals[i])), 2)) + (ma.pow((np.float(self.imags[i])), 2))))**2)
            N_I[i,i] = -1/((np.sqrt((ma.pow((np.float(self.reals[i])), 2)) + (ma.pow((np.float(self.imags[i])), 2))))**2)

        #print(N_R)
        #print(N_I)

        # Solving for x_coeff with Least Squares Estimation
        A_trans = [list(i) for i in zip(*A)]
        A_trans = np.array(A_trans)
        N_inv_R = np.linalg.pinv(N_R)
        N_inv_I = np.linalg.pinv(N_I)

        gain_coeff = (np.dot((np.linalg.pinv(np.dot((np.dot(A_trans, N_inv_R)), A))),
                             (np.dot((np.dot(A_trans, N_inv_R)), self.real_data))))
        phase_coeff = (np.dot((np.linalg.pinv(np.dot((np.dot(A_trans, N_inv_I)), A))),
                               (np.dot((np.dot(A_trans, N_inv_I)), self.imag_data))))

        #gain_coeff=(np.dot((np.dot((np.dot((np.linalg.pinv(np.dot((np.dot(A_trans,N_inv_R)),A))),A_trans)),N_inv_R)),self.real_data))
        #phase_coeff=(np.dot((np.dot((np.dot((np.linalg.pinv(np.dot((np.dot(A_trans,N_inv_I)),A))),A_trans)),N_inv_I)),self.imag_data))

        gains = gain_coeff[0:ant_no]
        minimum_pv = np.round(np.min(gains), 4) 
        maximum_pv = np.round(np.max(gains), 4) 
        #minimum_nv = np.round(np.min(gains), 4)
        #maximum_nv = 0
        if minimum_pv == maximum_pv:
            gains = gains/gains
        if minimum_pv != maximum_pv:
            for i in range(len(gains)):
                if gains[i] >= 0:
                    gains[i] = (np.float(gains[i] - minimum_pv) / np.float(maximum_pv - minimum_pv)) * 2
        #        if gains[i] < 0:
        #            gains[i] = (np.float(gains[i] - minimum_nv) / np.float(maximum_nv - minimum_nv))
        self.gain_coeff = gains

        self.phase_coeff = phase_coeff[0:ant_no]

        print("Gain_coeff " + str(self.gain_coeff))
        print("Phase_coeff " + str(self.phase_coeff))

    def parse_coeff(self, filename1, filename2):

        text_file = open(filename1, "w")
        for i in range(len(self.gain_coeff)):
            text_file.write(str(i) + ' ' + str(self.gain_coeff[i]) + '\n')
        text_file.close()

        text_file = open(filename2, "w")
        for i in range(len(self.phase_coeff)):
             text_file.write(str(i) + ' ' + str(self.phase_coeff[i]) + '\n')
        text_file.close()

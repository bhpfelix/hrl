# Hidden Markov Model Implementation

import pylab as pyl
import numpy as np
import matplotlib.pyplot as pp
#from enthought.mayavi import mlab

import scipy as scp
import scipy.ndimage as ni
from scipy.signal import butter, lfilter, freqz

# import roslib; roslib.load_manifest('sandbox_tapo_darpa_m3')
# import rospy
# import hrl_lib.mayavi2_util as mu
# import hrl_lib.viz as hv
# import hrl_lib.util as ut
# import hrl_lib.matplotlib_util as mpu
import pickle

import unittest
# import ghmm
# import ghmmwrapper
import random

import os, os.path

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b,a,data)
    return y

# Define features
# Input: Zt = (time_list, temp_list)
# Output: Fvec = [data] + [slope]
def feature_vector_diff(Zt,i=0):
    print 'Generating Fvec'
    temp_data = np.array(Zt[1][i:]).flatten().tolist()
    # last_value = temp_data[-1]
    # if len(temp_data) >= 500:
    #     temp_data = temp_data[0:500]
    # else:
    #     for i in range(len(temp_data), 500):
    #         temp_data.append(last_value)

    print 'Calculating Slope'
    print np.size(temp_data)
    temp_slope = []
    for j in range(np.size(temp_data)):
        if j <= 1 or j >= (np.size(temp_data)-1):
            temp_slope.append(0)
        else:
            #print ref_data[j+5], ref_data[j-5], ref_data[j+5]-ref_data[j-5]
            temp_slope.append((temp_data[j+1]-temp_data[j-1])/(2*0.01))

    order = 8 # 5?
    fs = 100.0
    cutoff = 2

    # Filter the data
    print 'Filtering Data'
    temp_slope = butter_lowpass_filter(np.array(temp_slope), cutoff, fs, order).tolist()

    return np.array([temp_data, temp_slope])


# if __name__ == '__main__' or __name__ != '__main__':

#     min_e = 0.089553894 #Rigid Polymer Foam
#     max_e = 107.8818103 #Copper
#     intervals_e = 22
#     delta_e = (max_e - min_e)/intervals_e
#     e_list = np.linspace(min_e, max_e, intervals_e)

#     data_path = 'data/'
#     exp_list = [(e_list[i], e_list[i+1]) for i in range(0, len(e_list)-1)]
#     exps = 1 #500

#     ta_dins = [0.0]*10000

#     ## Trials with different Initial Conditions
#     temp_num_dins = 0
#     for i in range(np.size(exp_list)):
#         path = data_path + str(i)
#         #print path
#         for trial in range(1, exps+1):

#             #print path + 'trial_' + np.str(num_file) + '.pkl'
#             ta_dins[exps*i + num_file - 1] = ut.load_pickle(path + 'trial_' + np.str(num_file) + '.pkl')
#         temp_num_dins = exps*np.size(exp_list)

#     Fmat_original = [0.0]*temp_num_dins

# ## Creating Feature Vector for DINS

#     idx = 0
#     while (idx < temp_num_dins):
#         Fmat_original[idx] = feature_vector_diff(ta_dins[idx],0)
#         idx = idx + 1

#     #print len(Fmat_original)
#     #print Fmat_original[85][:]
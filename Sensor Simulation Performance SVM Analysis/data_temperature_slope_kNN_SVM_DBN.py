# Hidden Markov Model Implementation

import pylab as pyl
import numpy as np
import matplotlib.pyplot as pp
#from enthought.mayavi import mlab

import scipy as scp
import scipy.ndimage as ni
from scipy.signal import butter, lfilter, freqz

import roslib; roslib.load_manifest('sandbox_tapo_darpa_m3')
import rospy
#import hrl_lib.mayavi2_util as mu
import hrl_lib.viz as hv
import hrl_lib.util as ut
import hrl_lib.matplotlib_util as mpu
import pickle

import unittest
import ghmm
import ghmmwrapper
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

def feature_vector_diff(Zt,i): 

    Fvec = Zt[1][i:-1]
    last_value = Fvec[-1]
    if len(Fvec) >= 500:
        Fvec = Fvec[0:500]
    else:
        for i in range(len(Fvec), 500):
            Fvec.append(last_value)
    slope = []
    for j in range(np.size(Fvec)):
        if j <= (2-1) or j >= (np.size(Fvec)-1):
            slope.append(0)
        else:
            #print ref_data[j+5], ref_data[j-5], ref_data[j+5]-ref_data[j-5]
            slope.append((Fvec[j+1]-Fvec[j-1])/(2*0.01))
            
    order = 8
    fs = 100.0
    cutoff = 2
    # Filter the data
    slope = butter_lowpass_filter(np.array(slope), cutoff, fs, order).tolist()
    for j in range(np.size(slope)):
        Fvec.append(slope[j])
    return Fvec

  
if __name__ == '__main__' or __name__ != '__main__':

    data_path = '/home/tapo/svn/robot1_data/usr/tapo/data/temperature_related/Automated_Random_Initial_Conditions/'
    exp_list = ['Acrylic/', 'Aluminum/', 'Brick/', 'Cardboard/', 'Glass/', 'MDF/', 'Neoprene/', 'Pine/', 'Porcelain/', 'Rubber/', 'Steel/']    
    exps = 500

    ta_dins = [0.0]*10000
            
## Trials with different Initial Conditions
    temp_num_dins = 0
    for i in range(np.size(exp_list)):
        path = data_path + exp_list[i]
        #print path
        for num_file in range(1, exps+1):
            #print path + 'trial_' + np.str(num_file) + '.pkl'
            ta_dins[exps*i + num_file - 1] = ut.load_pickle(path + 'trial_' + np.str(num_file) + '.pkl')
        temp_num_dins = exps*np.size(exp_list)

    Fmat_original = [0.0]*temp_num_dins
    
## Creating Feature Vector for DINS

    idx = 0
    while (idx < temp_num_dins):
        Fmat_original[idx] = feature_vector_diff(ta_dins[idx],0)
        idx = idx + 1

    #print len(Fmat_original)
    #print Fmat_original[85][:]

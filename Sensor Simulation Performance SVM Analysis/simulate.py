#!/usr/bin/env python

import matplotlib
import numpy as np
import matplotlib.pyplot as pp
import scipy as scp
import pickle
import optparse
import unittest
import random
import pickle
import os

from progressbar import *
import util
from forward_model import *
from data_temperature_slope_kNN_SVM_DBN import *
# from run_SVM_PCA import *


### File I/O
data_path = '../simulated_sensor_data/'

### For Effusivity Data Generation
min_e = 0.089553894 #Rigid Polymer Foam
max_e = 107.8818103 #Copper
intervals_e = 500
e_list = np.linspace(min_e, max_e, intervals_e)

### For Sensor Model
total_time = [0.25, 0.5, 1.0, 2, 4]
sampling_time = 0.005
k_sens = 0.0349
alpha_sens = 2.796*10**(-9)
t_sens_0 = [5, 10]
t_amb = [25]
k_obj = None # Value assigned later through sampling
alpha_obj = 1 # Given alpha_obj = 1, effusivity = k_obj/sqrt(alpha_obj) = k_obj
noise = [0, 0.01, 0.05, 0.1]

### For Simulated Experiment
exps = 20
# effu_interval_list = [(e_list[i], e_list[i+1]) for i in range(0, len(e_list)-1)]
# util.save_pickle(effu_interval_list, os.path.join(data_path, 'elist.pkl'))

### For Data Preprocessing During Experiment
fs = 100.0
cutoff = 2

### Creating Models
temp_models = [model_temperature(ts+ta, ta, max(total_time), sampling_time, k_sens, alpha_sens, k_obj, alpha_obj, n)
                    for ts in t_sens_0
                    for ta in t_amb
                    for n in noise]

for ind, model in enumerate(temp_models):
    print 'Iterating through model %s/%s' % (ind+1, len(temp_models))

    for trial in range(exps):
        print 'Iterating through trial %s/%s' % (trial+1, exps)

        widgets = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<')]
        pbar = ProgressBar(widgets=widgets, maxval=len(e_list)).start()

        Fmat = []
        for i, e in enumerate(e_list):
            model.set_attr('k_obj',e)

            result = model.run_simulation()
            Fvec = feature_vector_diff(result)
            Fmat.append(Fvec)

            pbar.update(i)


        fname = '%s_%s_%s_%s.pkl' % (str(trial).zfill(3),
                                model.t_sens_0,
                                model.t_ambient,
                                model.noise)

        pbar.finish()
        print

        print 'Saving Data'
        util.save_pickle(Fmat, os.path.join(data_path, fname))


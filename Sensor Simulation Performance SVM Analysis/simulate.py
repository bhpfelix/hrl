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

import util
from forward_model import *
from data_temperature_slope_kNN_SVM_DBN import *
# from run_SVM_PCA import *


### File I/O
data_path = 'data/'

### For Effusivity Data Generation
min_e = 0.089553894 #Rigid Polymer Foam
max_e = 107.8818103 #Copper
intervals_e = 3 #22
delta_e = (max_e - min_e)/intervals_e
e_list = np.linspace(min_e, max_e, intervals_e)

### For Sensor Model
total_time = [0.25, 0.5, 1, 1.5, 2, 3.5, 10]
sampling_time = 0.005
k_sens = 0.0349
alpha_sens = 2.796*10**(-9)
t_sens_0 = [30, 35, 40]
t_amb = [25] #[25, 30, 35]
k_obj = None # Value assigned later through sampling
alpha_obj = 1 # Given alpha_obj = 1, effusivity = k_obj/sqrt(alpha_obj) = k_obj
noise = [0.1] #[1, 2, 5, 10]

### For Simulated Experiment
exps = 1 #500
effu_interval_list = [(e_list[i], e_list[i+1]) for i in range(0, len(e_list)-1)]
util.save_pickle(effu_interval_list, os.path.join(data_path, 'elist.pkl'))

### For Data Preprocessing During Experiment
fs = 100.0
cutoff = 2


temp_models = [model_temperature(ts, ta, max(total_time), sampling_time, k_sens, alpha_sens, k_obj, alpha_obj, n)
                    for ts in t_sens_0
                    for ta in t_amb
                    for n in noise
                    if ta < ts]

for index, e_range in enumerate(effu_interval_list):
    print 'Iterating through range %s' % index
    for ind, model in enumerate(temp_models):
        print 'Iterating through model %s' % ind
        Fmat = []
        for trial in range(0, exps):
            print 'Trial #%s' % trial
            e_min, e_max = e_range
            normal_rand = np.random.randn()*0.7
            e = (e_min+e_max)/2. + normal_rand*(e_max-e_min)/2.

            model.set_attr('k_obj',e)

            result = model.run_simulation()
            Fvec = feature_vector_diff(result)
            Fmat.append(Fvec)

        fname = '%s_%s_%s_%s_%s.pkl' % (index,
                                        model.t_sens_0,
                                        model.t_ambient,
                                        model.sampling_time,
                                        model.noise)
        print 'Saving Data'
        util.save_pickle(Fmat, os.path.join(data_path, fname))







# data = [model.run_simulation() for model in temp_models]

# for index, item in enumerate(data):
#     time, temp_values = item

#     # filtered = butter_lowpass_filter(temp_values, cutoff, fs)
#     # temp_models[index].visualize_temp(time, filtered)
#     filtered_slope = feature_vector_diff(item)
#     temp_models[index].visualize_temp(time, filtered_slope[0])
#     temp_models[index].visualize_temp(time, filtered_slope[1])


# pp.show()
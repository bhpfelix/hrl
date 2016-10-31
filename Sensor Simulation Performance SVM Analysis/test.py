import matplotlib
import numpy as np
import matplotlib.pyplot as pp
import scipy as scp
import pickle
import optparse
import unittest
import random
import pickle

from forward_model import *
# from data_temperature_slope_kNN_SVM_DBN import *
# from run_SVM_PCA import *

total_time = [10] #[0.25, 0.5, 1, 1.5, 2, 3.5]
k_sens = 0.0349
alpha_sens = 2.796*10**(-9)
t_sens_0 = [30] #[30, 35, 40]
t_amb = [25] #[25, 30, 35]
k_obj = 0.15
alpha_obj = 0.15/(440.*1660.)
noise = [0.1] #[1, 2, 5, 10]

temp_models = [model_temperature(ts, ta, tt, k_sens, alpha_sens, k_obj, alpha_obj, n) for tt in total_time for ts in t_sens_0 for ta in t_amb for n in noise if ta < ts]
data = [model.run_simulation() for model in temp_models]
for index, item in enumerate(data):
    time, temp_values = item
    temp_models[index].visualize_temp(time, temp_values)

pp.show()
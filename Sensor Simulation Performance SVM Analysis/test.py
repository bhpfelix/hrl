import matplotlib
import numpy as np
import matplotlib.pyplot as pp
import scipy as scp
import pickle
import optparse
import unittest
import random
import pickle
import util
import os
from forward_model import *

ppp = util.load_pickle('data/0_308.15_298.15_0.005_0.001.pkl')
print np.array(ppp).shape

total_time = 10
sampling_time = 0.005
#from identify_sensor_parameters import k_sens, alpha_sens
k_sens = 0.0349
alpha_sens = 2.796*10**(-9)
t_sens_0 = 30
t_amb = 25
k_obj = 0.15
alpha_obj = 0.15/(440.*1660.)
noise = 0.1 #Percent

t_model = model_temperature(t_sens_0, t_amb, total_time, sampling_time, k_sens, alpha_sens, k_obj, alpha_obj, noise)
time_list = np.arange(0.01,total_time,sampling_time)

print len(ppp[0][0])
t_model.visualize_temp(time_list, ppp[0][0])
t_model.visualize_temp(time_list, ppp[0][1])
pp.show()

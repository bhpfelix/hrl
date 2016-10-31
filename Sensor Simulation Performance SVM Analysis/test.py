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
from data_temperature_slope_kNN_SVM_DBN import *
from run_SVM_PCA import *

k_sens = 0.0349
alpha_sens = 2.796*10**(-9)
t_sens_0 = 30
t_amb = 25
k_obj = 0.15
alpha_obj = 0.15/(440.*1660.)

temp_models = model_temperature(t_sens_0, t_amb, k_sens, alpha_sens, k_obj, alpha_obj)
time, temp_values = temp_models.run_simulation()
temp_models.visualize_temp(time, temp_values)

pp.show()
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

print np.array([[1,2],[3,4]]).flatten().tolist()
print [(1.*2.), (3.*4.)]
# data = np.linspace(1, 10, 5)
# util.save_pickle(data, 'data/test.1.0.pkl')

# data_load = util.load_pickle('data/test.1.0.pkl')
# print data_load
# print type(data_load)
dic = {(1,2):'ha', (3,4):'hei'}
print dic[(3,4)]

e_list = [1, 2, 3, 4]
print [(e_list[i], e_list[i+1]) for i in range(0, len(e_list)-1)]

print os.path.join('data/', str(1), 'test.pkl')

class tt:
    def __init__(self, t):
        self.t = t

    def set_val(self, attr, val):
        if hasattr(self, attr):
            exec('self.%s=val' % attr)
        else:
            raise AttributeError("'%s' object has no attribute '%s'" % (tt.__name__, attr))

ll = [0.7*np.random.randn() for i in range(100)]
print ll
print len([i for i in ll if i >= -1 and i <= 1])

z = ([1,2,3], [3,4,5])
print np.array(z[1][0:]).flatten().tolist()

ppp = util.load_pickle('data/0_303.15_298.15_0.005_0.001.pkl')
print len(ppp)

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

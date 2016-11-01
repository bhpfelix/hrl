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


print np.array([[1,2],[3,4]]).flatten().tolist()
print [(1.*2.), (3.*4.)]
# data = np.linspace(1, 10, 5)
# util.save_pickle(data, 'data/test.pkl')

# data_load = util.load_pickle('data/test.pkl')
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

ll = [0.7*np.random.randn(1) for i in range(100)]
print len([i for i in ll if i >= -1 and i <= 1])
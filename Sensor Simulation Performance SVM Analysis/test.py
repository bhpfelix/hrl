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

# ppp = util.load_pickle('../simulated_sensor_data/000_303.15_298.15_0.1.pkl')
# print np.array(ppp).shape

score_1 = 2
score_2 = 1
mean_score = score_2 if score_1 < 0 else (score_1 if score_2 < 0 else (score_1+score_2)/2.)
k = (1,2,3)
print "%s%s%s" % k
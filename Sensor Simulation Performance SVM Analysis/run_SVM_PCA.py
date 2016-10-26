#!/usr/bin/env python

# SVM+PCA for Temperature Data
import matplotlib
import numpy as np
import matplotlib.pyplot as pp
import scipy as scp
import roslib; roslib.load_manifest('sandbox_tapo_darpa_m3')
import rospy
import hrl_lib.util as ut
import hrl_lib.matplotlib_util as mpu
import pickle
import optparse
import unittest
import random

from sklearn import decomposition
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import sys
#sys.path.insert(0, '/home/tapo/svn/robot1_data/usr/tapo/data_code/temperature_related/')
sys.path.insert(0, '/home/tapo/svn/robot1_data/usr/tapo/data_code/temperature_related/Automated_Random_Initial_Conditions/')
from data_temperature_slope_kNN_SVM_DBN import Fmat_original

def create_dataset(mat, categories):
    # Acrylic = 1, Aluminum = 2, Brick = 3, Cardboard = 4, Glass = 5, MDF = 6, Neoprene = 7, Pine = 8, Porcelain = 9, Rubber = 10, Steel = 11
    local_dataset = {'data':None, 'target':[]}
    local_dataset['data'] = np.array(mat)
    for i in range(np.shape(local_dataset['data'])[0]):
        local_dataset['target'].append(1 if (i < 500) else 2 if (i >= 500 and i < 1000) else 3 if (i >= 1000 and i < 1500) else 4 if (i >= 1500 and i < 2000) else 5 if (i >= 2000 and i < 2500) else 6 if (i >= 2500 and i < 3000) else 7 if (i >= 3000 and i < 3500) else 8 if (i >= 3500 and i < 4000) else 9 if (i >= 4000 and i < 4500) else 10 if (i >= 4500 and i < 5000) else 11)
    local_dataset['target'] = np.array(local_dataset['target'])
    return local_dataset

def run_pca(dataset):
    pca = decomposition.PCA(n_components=10)
    pca.fit(dataset['data'])
    reduced_mat = pca.transform(dataset['data'])
    dataset['data'] = reduced_mat
    return dataset

def run_crossvalidation(data_dict, categories, folds):
    data_dict = run_pca(data_dict)
    skf = StratifiedKFold(data_dict['target'], n_folds=folds)
    confusion_matrix_final = np.zeros((np.size(categories), np.size(categories)))
    for train, test in skf:
        X_train, X_test, y_train, y_test = data_dict['data'][train], data_dict['data'][test], data_dict['target'][train], data_dict['target'][test]
        svc = svm.SVC(kernel='linear')
        #svc = svm.SVC(kernel='rbf')
        #svc = svm.SVC(kernel='poly', degree=3)
        svc.fit(X_train, y_train)
        preds = svc.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        print np.shape(preds)
        print np.shape(y_test)
        print '############################################################'
        print classification_report(y_test, preds)
        print "Confusion matrix: \n%s" %confusion_matrix(y_test, preds)
        confusion_matrix_final = confusion_matrix_final + cm
    # Show confusion matrix in a separate window
	pp.matshow(confusion_matrix_final)
	pp.title('Confusion matrix')
	pp.colorbar()
	pp.ylabel('True label')
	pp.xlabel('Predicted label')
	pp.show()

def run_crossvalidation_new(data_dict, categories, folds):
    data_dict = run_pca(data_dict)
    skf = StratifiedKFold(data_dict['target'], n_folds=folds)
    svc = svm.SVC(kernel='linear')
    #svc = svm.SVC(kernel='rbf')
    #svc = svm.SVC(kernel='poly', degree=3)
    scores = cross_val_score(svc, data_dict['data'], data_dict['target'], cv=skf)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()/2)


if __name__ == '__main__':

    p = optparse.OptionParser()
    p.add_option('--raw', action='store_true', dest='raw', help='use raw features')
    p.add_option('--raw_slope', action='store_true', dest='raw_slope', help='use raw and slope features')
        
    opt, args = p.parse_args()

    if opt.raw:
        input_Fmat = np.array(Fmat_original)[:,0:500].tolist()
    elif opt.raw_slope:
        input_Fmat = Fmat_original
    else:
        sys.exit("Please specify --raw or --raw_slope")

    categories = ['Acrylic', 'Aluminum', 'Brick', 'Cardboard', 'Glass', 'MDF', 'Neoprene', 'Pine', 'Porcelain', 'Rubber', 'Steel']
    num_folds = 3
    #print np.shape(input_Fmat)
    dataset = {}  
    dataset = create_dataset(input_Fmat, categories)
    run_crossvalidation(dataset, categories, num_folds)
    run_crossvalidation_new(dataset, categories, num_folds)
    
    
    
    


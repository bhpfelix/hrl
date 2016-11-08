#!/usr/bin/env python

# SVM+PCA for Temperature Data
import matplotlib
import numpy as np
import matplotlib.pyplot as pp
import scipy as scp
from math import floor
import pickle
import optparse
import unittest
import random
import os
import util

from sklearn import decomposition
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import sys

plt_path = "../simulation_data_plots"
mat_path = "../simulation_confusion_mats"
data_path = "../simulated_sensor_data"
svm_path = "../trained_svms"
pred_path = "../predictions"
matsize = 420
exp = 20
num_folds = 3
total_time = [1.0, 2.4, 3.8, 5.2, 6.6, 8]
MAX_TIME = max(total_time)

if not os.path.exists(mat_path):
    os.makedirs(mat_path)
if not os.path.exists(plt_path):
    os.makedirs(plt_path)
if not os.path.exists(svm_path):
    os.makedirs(svm_path)
if not os.path.exists(pred_path):
    os.makedirs(pred_path)
for i in total_time:
    if not os.path.exists(os.path.join(plt_path,'%0.2f' % i)):
        os.makedirs(os.path.join(plt_path,'%0.2f' % i))
    if not os.path.exists(os.path.join(svm_path,'%0.2f' % i)):
        os.makedirs(os.path.join(svm_path,'%0.2f' % i))
    if not os.path.exists(os.path.join(pred_path,'%0.2f' % i)):
        os.makedirs(os.path.join(pred_path,'%0.2f' % i))


def create_dataset(mat, index, t_sens, t_amb, noise, dataset):
    for i in range(0, len(mat)):
        key = (t_sens, t_amb, noise)
        if key not in dataset:
            dataset[key] = {'data': [], 'target': []}
        dataset[key]['data'] += [np.array(mat[i])]
        dataset[key]['target'] += [index]

def run_pca(dataset):
    # pca = decomposition.PCA(n_components=10)
    pca = decomposition.PCA()
    pca.fit(dataset['data'])
    reduced_mat = pca.transform(dataset['data'])
    dataset['data'] = reduced_mat
    return dataset

def run_crossvalidation(mat, key, data_dict, folds, total_time):
    t_sens, t_amb, noise = key
    mat_key = (t_sens, t_amb, noise, total_time)

    print 'Run PCA'
    d = run_pca(data_dict)
    skf = StratifiedKFold(d['target'], n_folds=folds, shuffle=True)
    confusion_matrix_final = np.zeros((matsize, matsize))

    for model_num, fold in enumerate(skf):
        train, test = fold
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for t in train:
            X_train.append(d['data'][t])
            y_train.append(d['target'][t])
        for t in test:
            X_test.append(d['data'][t])
            y_test.append(d['target'][t])

        print 'Training'
        svc = svm.SVC(kernel='linear')
        #svc = svm.SVC(kernel='rbf')
        #svc = svm.SVC(kernel='poly', degree=3)
        svc.fit(X_train, y_train)
        util.save_pickle(svc, os.path.join(svm_path, '%.2f'%total_time, '%s_%s_%s_%s.pkl'%(model_num, t_sens, t_amb, noise)))

        preds = svc.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        util.save_pickle((y_test, preds), os.path.join(pred_path, '%.2f'%total_time, '%s_%s_%s_%s.pkl'%(model_num, t_sens, t_amb, noise)))

        print np.shape(preds)
        print np.shape(y_test)
        print '############################################################'
        print classification_report(y_test, preds)
        print "Confusion matrix: \n%s" %confusion_matrix(y_test, preds)
        confusion_matrix_final = confusion_matrix_final + cm

    # Show confusion matrix in a separate window
	pp.matshow(confusion_matrix_final)
	pp.title('Confusion matrix\n t_sens = %s, t_amb = %s\nnoise = %s, total_time = %.2f' %(t_sens, t_amb, noise, total_time))
	pp.colorbar()
	pp.ylabel('True label')
	pp.xlabel('Predicted label')
	#pp.show()
    pp.savefig('%s/%.2f/%s_%s_%s_%.2f.png' %(plt_path, total_time, t_sens, t_amb, noise, total_time))
    pp.close()
    matrices[mat_key] = confusion_matrix_final


def run_crossvalidation_new(data_dict, categories, folds):
    data_dict = run_pca(data_dict)
    skf = StratifiedKFold(data_dict['target'], n_folds=folds)
    svc = svm.SVC(kernel='linear')
    #svc = svm.SVC(kernel='rbf')
    #svc = svm.SVC(kernel='poly', degree=3)
    scores = cross_val_score(svc, data_dict['data'], data_dict['target'], cv=skf)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()/2)


if __name__ == '__main__':

    datatags = {}
    for f in os.listdir(data_path):
        if f.endswith(".pkl"):
            trial, t_sens, t_amb, noise = f[:-4].split('_')
            key = (t_sens, t_amb, noise)

            if key in datatags:
                datatags[key].append(f)
            else:
                datatags[key] = [f]

    print datatags

    for t in total_time:
        print 'Iterating through time %.2f' % t
        matrices = {}
        for k in datatags:
            print 'Iterating through model %s' % str(k)
            temp_data = {'data':[], 'target':[]}
            for fname in datatags[k]:
                dataVec = util.load_pickle(os.path.join(data_path, fname))
                for pos,fvec in enumerate(dataVec):
                    temp, slope = fvec
                    length = len(temp)
                    temp = temp[:int(t*length/MAX_TIME)]
                    slope = slope[:int(t*length/MAX_TIME)]
                    slope[-1] = 0
                    temp_data['data'].append(temp + slope)
                    temp_data['target'].append(str(pos))

            run_crossvalidation(matrices, k, temp_data, num_folds, t)
        util.save_pickle(matrices, os.path.join(mat_path, 'confusion_matrices_%.2f.pkl' %t))
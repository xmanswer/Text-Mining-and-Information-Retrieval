# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:49:23 2016

@author: minxu

This script is used to conduct experiment and generate predicting results
for movie recommendation from a subset of Netflix dataset, locate this script
with other python files as well as data set files in the same folder. Intermediate
.npy files may be generated.  

Two switches can be enabled in order to conduct development experiment (__dev__)
and generate final test set predictions (__test__)

"""

import preprocess
import RMLR
import PMF
import math
import os
import time
from sklearn import svm
import numpy as np
import scipy as sp
from scipy import sparse
import eval_ndcg_mod

__dev__ = False
__test__ = True

userSize = preprocess.userSize
movieSize = preprocess.movieSize


def predict_score(classifier, U, V, test_userlist, test_umdict):
    um_score_dict = dict()
    for u in test_umdict:
        score_dict = dict()
        movies = test_umdict[u]
        movie_vectors = dict()
        for m in movies:
            movie_vectors[m] = np.multiply(U[:, u], V[:, m])
            score_dict[m] = 0
            
        for m1 in movies:
            for m2 in movies:
                if m1 != m2:
                    v = movie_vectors[m1] - movie_vectors[m2]
                    score_dict[m1] = score_dict[m1] + classifier.predict(v.transpose())[0]
        
        um_score_dict[u] = score_dict
    
    return um_score_dict


def statistic(I, train_class):
    print 'total number of observed ratings is ' + str(I.sum())
    print 'total number of training samples in T is ' + str(train_class.shape[0])
    pos = float(train_class.sum())
    neg = float(train_class.shape[0] - pos)
    print 'ratio of pos samples to neg is ' + str(pos/neg)
    
    

R, I = preprocess.parseTrainingSet()
R_transpose = R.transpose()
I_transpose = I.transpose()


if __dev__:
    D = [10, 20, 50, 100]
    lr_res = []
    svm_res = []
    
    for d in D:
        trainset_PMF = PMF.PMF(R, R_transpose, I, I_transpose, D = d)
        if not os.path.exists('dev_umdict_' + str(d) + '.npy'):        
            train_matrix, train_class = trainset_PMF.feature_construction()
            np.save('train_matrix_' + str(d) + '.npy', train_matrix)
            np.save('train_class_' + str(d) + '.npy', train_class)
        else:
            train_matrix = np.load('train_matrix_' + str(d) + '.npy') 
            train_class = np.load('train_class_' + str(d) + '.npy')
            
        dev_umdict, dev_userlist, dev_movielist = preprocess.parseDevTestSet('dev.csv')
        
        statistic(I, train_class)    
        
        t = time.time()
        rmlr = RMLR.RMLR_classifier(lam = 0.1, maxIter = 500)
        rmlr.fit(train_matrix, train_class)
        print 'lr-letor run time for d = ' + str(d) + ' is: ' + str(time.time() - t)
        dev_score_LR = predict_score(rmlr, trainset_PMF.U, trainset_PMF.V, dev_userlist, dev_umdict)
        preprocess.write_result('dev_class_LR_' + str(d), dev_score_LR, dev_userlist, dev_movielist)
        res = eval_ndcg_mod.run('dev.csv', 'dev_class_LR_' + str(d), 'dev.golden')
        lr_res.append(res)
        
        
        t = time.time()
        svm_classifier = svm.LinearSVC(C = 1e-6, max_iter = 500)
        svm_classifier.fit(train_matrix, train_class)
        print 'svm run time for d = ' + str(d) + ' is: ' + str(time.time() - t) 
        dev_score_SVM = predict_score(svm_classifier, trainset_PMF.U, trainset_PMF.V, dev_userlist, dev_umdict)
        preprocess.write_result('dev_class_SVM_' + str(d), dev_score_SVM, dev_userlist, dev_movielist)
        res = eval_ndcg_mod.run('dev.csv', 'dev_class_SVM_' + str(d), 'dev.golden')
        svm_res.append(res)

if __test__:
    d = 50
    trainset_PMF = PMF.PMF(R, R_transpose, I, I_transpose, D = d)
    if not os.path.exists('train_matrix_' + str(d) + '.npy'):
        train_matrix, train_class = trainset_PMF.feature_construction()
        np.save('train_matrix_' + str(d) + '.npy', train_matrix)
        np.save('train_class_' + str(d) + '.npy', train_class)
    else:
        train_matrix = np.load('train_matrix_' + str(d) + '.npy') 
        train_class = np.load('train_class_' + str(d) + '.npy')    
        
    test_umdict, test_userlist, test_movielist = preprocess.parseDevTestSet('test.csv')
    svm_classifier = svm.LinearSVC(C = 1e-6, max_iter = 500)
    svm_classifier.fit(train_matrix, train_class)
    test_score_SVM = predict_score(svm_classifier, trainset_PMF.U, trainset_PMF.V, test_userlist, test_umdict)
    preprocess.write_result('predictions.txt', test_score_SVM, test_userlist, test_movielist)
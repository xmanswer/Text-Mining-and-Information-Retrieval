# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:49:23 2016

This is the main script of this assignment. Put this file and others in the same
folder as the original data. This will call necessary scripts to parse the data.
Notice that it will generate some temporary files for the data processing part. 
The purpose of this is to accelerate the experiment speed so that I do not have
to parse the raw dataset every time I do some experiment. The temporary files 
have the size of around 400 MB. 

The experiment switches can be modified to conduct certain experiement. By default
only the tfidf one is turned on since it gives the best results.
For each experiemnt, it will generate result files, with one stores the training
accuracy, the other one stores the prediction of the scores (with the name of the
format like dev_trainAndClassify_$CLASSIFIERNAME_$FEATURENAME_$LAMBDA)

For RMLR, the script will call train_RMLR and classify_RMLR to do training and
classification; for SVM, the script will call sklearn.svm.LinearSVC to do the
same task. So please make sure the python environment has the necassary package
such as scipy and sklearn.

@author: minxu
"""

import preprocess
import train_RMLR
import classify_RMLR
import math
from sklearn import svm
import numpy as np

#==============================================================================
# experiment switches
#==============================================================================
DO_RMLR_CTF = False
DO_RMLR_DF = False
DO_SVM_CTF = False
DO_SVM_DF = False
DO_RMLR_TFIDF = True

#==============================================================================
# trainning accuracy calculation
#==============================================================================
def training_accuracy_hard(score, prediction):
    return (score == prediction).sum(0) / float(train_d_size)

def training_rmse_soft(score, prediction):
    diffmatrix = score - prediction
    return math.sqrt(diffmatrix.transpose().dot(diffmatrix) / float(train_d_size))

#==============================================================================
# generate dev classification file
#==============================================================================
def generateClassFile(filename, hard_class, soft_class):
    n = hard_class.shape[0]
    with open(filename, 'w') as f:
        for i in range(0, n):
            f.write(str(int(hard_class[i])) + ' ' + str(soft_class[i]) + '\n')            

def generateTrainAccuracyFile(filename, train_accuracy_hard, train_accuracy_soft):
    with open(filename, 'w') as f:
        for i in range(0, len(train_accuracy_hard)):
            f.write(str(train_accuracy_hard[i]) + ' ' + str(train_accuracy_soft[i]) + '\n')  

#==============================================================================
# do train and classifications, save dev class file and train accuracy file
# based on the input classification method
#==============================================================================
def trainAndClassify(trainAndClassifyMethod, feature_type, train_score, dev_Matrix, train_Matrix, Lambdas):
    train_accuracy_hard = []
    train_accuracy_soft = []
    for lam in Lambdas:
        print "do lambda = " + str(lam)
        dev_hard_class, dev_soft_class, train_hard_class, train_soft_class = trainAndClassifyMethod(train_score, dev_Matrix, train_Matrix, lam, train_accuracy_hard, train_accuracy_soft)
        generateClassFile('dev_' + trainAndClassifyMethod.func_name + '_' + feature_type + '_' + str(lam), dev_hard_class, dev_soft_class)   
    
    generateTrainAccuracyFile('train_' + trainAndClassifyMethod.func_name + '_' + feature_type, train_accuracy_hard, train_accuracy_soft)
    return dev_hard_class, dev_soft_class, train_hard_class, train_soft_class

#==============================================================================
# train and classification using RMLR
#==============================================================================
def trainAndClassify_RMLR(train_score, dev_Matrix, train_Matrix, lam, train_accuracy_hard, train_accuracy_soft):
    w = train_RMLR.gradientAsc(train_Matrix, train_score, lam, lr, error, maxIter, train_RMLR.allBatch)
    train_hard_class = classify_RMLR.classify_hard(train_Matrix, w)
    train_soft_class = classify_RMLR.classify_soft(train_Matrix, w)
    dev_hard_class = classify_RMLR.classify_hard(dev_Matrix, w)
    dev_soft_class = classify_RMLR.classify_soft(dev_Matrix, w)
    
    train_error_hard = training_accuracy_hard(np.array(train_score).T[0], train_hard_class)
    train_error_soft = training_rmse_soft(np.array(train_score).T[0], train_soft_class)
    
    print train_error_hard
    train_accuracy_hard.append(train_error_hard)
    print train_error_soft
    train_accuracy_soft.append(train_error_soft)
    
    return dev_hard_class, dev_soft_class, train_hard_class, train_soft_class

#==============================================================================
# train and classification using SVM
#==============================================================================
def trainAndClassify_SVM(train_score, dev_Matrix, train_Matrix, lam, train_accuracy_hard, train_accuracy_soft):
    svm_classifier = svm.LinearSVC(C = lam, max_iter = maxIter)
    svm_classifier.fit(train_Matrix, np.array(train_score))
    train_hard_class = svm_classifier.predict(train_Matrix)
    dev_hard_class = svm_classifier.predict(dev_Matrix)

    train_error_hard = training_accuracy_hard(np.array(train_score).T[0], train_hard_class)
    print train_error_hard
    train_accuracy_hard.append(train_error_hard)
    train_accuracy_soft.append(train_error_hard)
    
    return dev_hard_class, dev_hard_class, train_hard_class, train_hard_class
    
#==============================================================================
# main script starts
#==============================================================================
#process necessary data structures for learning and classification
tokenDict_ctf, tokenDict_df, devdata, testdata = preprocess.main()
dev_dwMatrix_ctf, dev_dfVector_ctf, dev_ctfVector_ctf = preprocess.featureExtraction_test(devdata, tokenDict_ctf)    
test_dwMatrix_ctf, test_dfVector_ctf, test_ctfVector_ctf = preprocess.featureExtraction_test(testdata, tokenDict_ctf)    
train_score_ctf, train_dwMatrix_ctf, train_dfVector_ctf, train_ctfVector_ctf = preprocess.featureExtraction('train_preprocessed_ctf')
dev_dwMatrix_df, dev_dfVector_df, dev_ctfVector_df = preprocess.featureExtraction_test(devdata, tokenDict_df)
train_score_df, train_dwMatrix_df, train_dfVector_df, train_ctfVector_df = preprocess.featureExtraction('train_preprocessed_df')


#RMLR parameters
train_d_size = preprocess.train_d_size
w_size = preprocess.w_size
Lambdas = [1] #this also serves as C in SVM
lr = 0.0000002 #learning rate
error = 1e-9 #error for RMLR
maxIter = 2000 #max Interation number for both RMLR and SVM

#do RMLR on ctf based feature
if DO_RMLR_CTF:
    print "do ctf based feature learning and classification using rmlr"
    trainAndClassify(trainAndClassify_RMLR, 'ctf', train_score_ctf, dev_dwMatrix_ctf, train_dwMatrix_ctf, Lambdas)

#do RMLR on df based feature
if DO_RMLR_DF:
    print "do df based feature learning and classification using rmlr"
    trainAndClassify(trainAndClassify_RMLR, 'df', train_score_df, dev_dwMatrix_df, train_dwMatrix_df, Lambdas)

#do SVM on ctf based feature
if DO_SVM_CTF:
    print "do ctf based feature learning and classification using svm"
    trainAndClassify(trainAndClassify_SVM, 'ctf', train_score_ctf, dev_dwMatrix_ctf, train_dwMatrix_ctf, Lambdas)
    
#do SVM on ctf based feature
if DO_SVM_DF:
    print "do df based feature learning and classification using svm"
    trainAndClassify(trainAndClassify_SVM, 'df', train_score_df, dev_dwMatrix_df, train_dwMatrix_df, Lambdas)

#do RMLR on tf-idf based feature
if DO_RMLR_TFIDF:
    print "do tf-idf based feature learning and classification using rmlr"
    train_score_tfidf = train_score_ctf
    dev_dwMatrix_tfidf = preprocess.featureExtraction_custom(dev_dwMatrix_ctf, dev_dfVector_ctf)
    train_dwMatrix_tfidf = preprocess.featureExtraction_custom(train_dwMatrix_ctf, train_dfVector_ctf)
    test_dwMatrix_tfidf = preprocess.featureExtraction_custom(test_dwMatrix_ctf, test_dfVector_ctf)

    trainAndClassify(trainAndClassify_RMLR, 'tfidf', train_score_tfidf, dev_dwMatrix_tfidf, train_dwMatrix_tfidf, Lambdas)
    trainAndClassify(trainAndClassify_RMLR, 'test', train_score_tfidf, test_dwMatrix_tfidf, train_dwMatrix_tfidf, Lambdas)

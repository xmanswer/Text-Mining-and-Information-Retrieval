# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:52:21 2016

This script provides functions to do hard and soft classification for RMLR
and given weight vector and input sparse matrix. 

@author: minxu
"""
import scipy as sp
from scipy import sparse
import numpy as np


#==============================================================================
# hard classification for RMLR
#==============================================================================
def classify_hard(test_matrix, w):
    predict_matrix = classify(test_matrix, w)    
    return np.array(predict_matrix.argmax(1) + 1).T[0]

#==============================================================================
# soft classification for RMLR
#==============================================================================
def classify_soft(test_matrix, w):
    predict_matrix = classify(test_matrix, w)
    predict = np.matrix(np.zeros((test_matrix.shape[0], 1)))
    for c in range(0, w.shape[1]):
        predict = predict + (c + 1) * predict_matrix[:, c]
    
    return np.array(predict).T[0]

#==============================================================================
# main classification routine, return a nxc matrix for use in 
# both hard and soft classifications
#==============================================================================
def classify(test_matrix, w):
    test_size, w_size = test_matrix.shape
    w_size, C = w.shape
    predict_matrix = np.matrix(np.zeros((test_size, C)))
    w_x = test_matrix.dot(w)
    for c in range(0, C):
        predict_matrix[:, c] = np.exp(w_x[:, c]) / np.exp(w_x).sum(1)
    
    return predict_matrix
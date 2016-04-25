# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 17:02:48 2016

This script provides functions to do training. It pack three different gradient
ascent methods (aka gradeint ascent, stochastic gradient ascent and 
batched-stochastic gradeint ascent). It also contains functions to calculate 
the graident and objective function for given sparse matrix and weight vectors.

@author: minxu
"""
import scipy as sp
from scipy import sparse
import numpy as np
import preprocess
import math

train_d_size = preprocess.train_d_size
#w_size = preprocess.w_size
C = 5
obj_subset_size = 50000

#==============================================================================
# perform gradient ascent on training set, return the learned parameter
# matrix w (f x c), notice that the convergence standards are based on both
# error and max interation number
# lam is the regularization parameter and lr is the learning rate
#==============================================================================
def gradientAsc(dwMatrix, score, lam, lr, error, maxIter, acsFunc):
    N, w_size = dwMatrix.shape
    w = np.matrix(np.zeros((w_size, C)))
    return acsFunc(dwMatrix, score, lam, lr, error, maxIter, w)

#==============================================================================
# update all samples at each iteration
#==============================================================================
def allBatch(dwMatrix, score, lam, lr, error, maxIter, w):
    N, w_size = dwMatrix.shape
    last_obj = obj(dwMatrix, score, w, lam)
    first_obj = last_obj
    for i in range(0, maxIter):
        new_w = np.matrix(np.zeros((w_size, C)))
        
        for c in range(0, C):
            new_w[:, c] = w[:, c] + lr * grad(dwMatrix, score, c, w, lam)
        w = new_w
        #print i
        if i % 1 == 0:
            l_w = obj(dwMatrix[1:obj_subset_size, :], score[1:obj_subset_size, :], w, lam)
            diff = math.fabs(l_w - last_obj) / math.fabs(l_w - first_obj)
            #print l_w
            #print diff
            last_obj = l_w
            if(diff < error): break
    
    return w

#==============================================================================
# update one sample at each iteration
#==============================================================================
def stochastic(dwMatrix, score, lam, lr, error, maxIter, w):
    N, w_size = dwMatrix.shape
    last_obj = obj(dwMatrix, score, w, lam)
    first_obj = last_obj
    for i in range(0, maxIter):
        for d in range(0, train_d_size):
            neww = np.matrix(np.zeros((w_size, C)))
            for c in range(0, C):
                neww[:, c] = w[:, c] + lr * grad(dwMatrix[d,:], score[d,:], c, w, lam)
            w = neww
            if d % 1 == 0:
                l_w = obj(dwMatrix[1:obj_subset_size, :], score[1:obj_subset_size, :], w, lam)
                diff = math.fabs(l_w - last_obj) / math.fabs(first_obj)
                last_obj = l_w
                if(diff < error): return w
    
    return w

#==============================================================================
# update a batch of samples on each iteration
#==============================================================================
def batch(dwMatrix, score, lam, lr, error, maxIter, w, batchSize = 100):
    N, w_size = dwMatrix.shape
    last_obj = obj(dwMatrix, score, w, lam)
    first_obj = last_obj
    for i in range(0, maxIter):
        for d in range(0, batchSize):
            s =  train_d_size * (d / batchSize)
            e = train_d_size * ((d + 1) / batchSize)
            neww = np.matrix(np.zeros((w_size, C)))
            for c in range(0, C):
                neww[:, c] = w[:, c] + lr * grad(dwMatrix[s:e, :], score[s:e, :], c, w, lam)
            w = neww
            if d % 1 == 0:
                l_w = obj(dwMatrix[1:obj_subset_size, :], score[1:obj_subset_size, :], w, lam)
                diff = math.fabs(l_w - last_obj) / math.fabs(first_obj)
                last_obj = l_w
                if(diff < error): return w
    
    return w
#==============================================================================
# calculate the gradient for given w
#==============================================================================
def grad(dwMatrix, score, c, w, lam):
    yc = score == (c + 1) #n x 1 binary label vector
    w_x = dwMatrix.dot(w) #n x c matrix where each element is 1x1 result of xi * wc
    gd = dwMatrix.transpose() * (yc - (np.exp(w_x[:, c]) / np.exp(w_x).sum(1))) - lam * w[:, c]
    #print gd
    return gd

#==============================================================================
# calculate the value of objective function
#==============================================================================
def obj(dwMatrix, score, w, lam):
    P_y_xw = 0 #log likelihood of P(y|x,W), a nx1 matrix
    regw = 0 #a scala store the sum of wTw given different class
    w_x = dwMatrix.dot(w) #n x c matrix where each element is 1x1 result of xi * wc
    for c in range(0, C):
        yc = score == (c + 1)
        P_y_xw = P_y_xw + np.multiply(yc, np.log((np.exp(w_x[:, c]) / np.exp(w_x).sum(1))))
        regw = regw + w[:,c].transpose().dot(w[:,c])
    
    l_w = P_y_xw.sum(0) - lam / 2 * regw
    return l_w
    
    

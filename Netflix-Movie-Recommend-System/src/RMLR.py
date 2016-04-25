# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 17:02:48 2016

This class provides functions to do training and classification. It pack three 
different gradient ascent methods (aka gradeint ascent, stochastic gradient ascent and 
batched-stochastic gradeint ascent). It also contains functions to calculate 
the graident and objective function for given sparse matrix and weight vectors.

@author: minxu
"""
import scipy as sp
from scipy import sparse
import numpy as np
import math

class RMLR_classifier:    
    
    train_d_size = 0
    w_size = 0
    C = 2
    obj_subset_size = 50000
    lam = 1 #reg term
    lr = 0.0000002 #learning rate
    error = 1e-9 #error for RMLR
    maxIter = 100 #max Interation number
    
    def __init__(self, C = None, lam = None, lr = None, error = None, maxIter = None):
        if C == None:
            C = self.C
        if lam == None:
            lam = self.lam
        if lr == None:
            lr = self.lr
        if error == None:
            error = self.error
        if maxIter == None:
            maxIter = self.maxIter
    
    def fit(self, train_matrix, train_score):
        self.train_d_size, self.w_size = train_matrix.shape
        self.w = self.gradient_asc(train_matrix, train_score, self.all_batch)
    
    #==============================================================================
    # perform gradient ascent on training set, return the learned parameter
    # matrix w (f x c), notice that the convergence standards are based on both
    # error and max interation number
    # lam is the regularization parameter and lr is the learning rate
    #==============================================================================
    def gradient_asc(self, matrix, score, acsFunc):
        w = np.matrix(np.zeros((self.w_size, self.C)))
        return acsFunc(matrix, score, w)
    
    #==============================================================================
    # update all samples at each iteration
    #==============================================================================
    def all_batch(self, matrix, score, w):
        last_obj = self.obj(matrix, score, w)
        first_obj = last_obj
        for i in range(0, self.maxIter):
            new_w = np.matrix(np.zeros((self.w_size, self.C)))
            
            for c in range(0, self.C):
                new_w[:, c] = w[:, c] + self.lr * self.grad(matrix, score, c, w)
            w = new_w
            #print i
            if i % 1 == 0:
                l_w = self.obj(matrix[1:self.obj_subset_size, :], score[1:self.obj_subset_size, :], w)
                diff = math.fabs(l_w - last_obj) / math.fabs(l_w - first_obj)
                #print l_w
                #print diff
                last_obj = l_w
                if(diff < self.error): break
        
        return w
    
    #==============================================================================
    # update one sample at each iteration
    #==============================================================================
    def stochastic(self, matrix, score, w):
        last_obj = self.obj(matrix, score, w)
        first_obj = last_obj
        for i in range(0, self.maxIter):
            for d in range(0, self.train_d_size):
                neww = np.matrix(np.zeros((self.w_size, self.C)))
                for c in range(0, self.C):
                    neww[:, c] = w[:, c] + self.lr * self.grad(matrix[d,:], score[d,:], c, w)
                w = neww
                if d % 1 == 0:
                    l_w = self.obj(matrix[1:self.obj_subset_size, :], score[1:self.obj_subset_size, :], w)
                    diff = math.fabs(l_w - last_obj) / math.fabs(first_obj)
                    last_obj = l_w
                    if(diff < self.error): return w
        
        return w
    
    #==============================================================================
    # update a batch of samples on each iteration
    #==============================================================================
    def batch(self, matrix, score, w, batchSize = 100):
        last_obj = self.obj(matrix, score, w)
        first_obj = last_obj
        for i in range(0, self.maxIter):
            for d in range(0, batchSize):
                s =  self.train_d_size * (d / batchSize)
                e = self.train_d_size * ((d + 1) / batchSize)
                neww = np.matrix(np.zeros((self.w_size, self.C)))
                for c in range(0, self.C):
                    neww[:, c] = w[:, c] + self.lr * self.grad(matrix[s:e, :], score[s:e, :], c, w)
                w = neww
                if d % 1 == 0:
                    l_w = self.obj(matrix[1:self.obj_subset_size, :], score[1:self.obj_subset_size, :], w)
                    diff = math.fabs(l_w - last_obj) / math.fabs(first_obj)
                    last_obj = l_w
                    if(diff < self.error): return w
        
        return w
    #==============================================================================
    # calculate the gradient for given w
    #==============================================================================
    def grad(self, matrix, score, c, w):
        yc = score == c #n x 1 binary label vector
        w_x = matrix.dot(w) #n x c matrix where each element is 1x1 result of xi * wc
        gd = matrix.transpose() * (yc - (np.exp(w_x[:, c]) / np.exp(w_x).sum(1))) - self.lam * w[:, c]
        #print gd
        return gd
    
    #==============================================================================
    # calculate the value of objective function
    #==============================================================================
    def obj(self, matrix, score, w):
        P_y_xw = 0 #log likelihood of P(y|x,W), a nx1 matrix
        regw = 0 #a scala store the sum of wTw given different class
        w_x = matrix.dot(w) #n x c matrix where each element is 1x1 result of xi * wc
        for c in range(0, self.C):
            yc = score == c
            P_y_xw = P_y_xw + np.multiply(yc, np.log((np.exp(w_x[:, c]) / np.exp(w_x).sum(1))))
            regw = regw + w[:,c].transpose().dot(w[:,c])
        
        l_w = P_y_xw.sum(0) - self.lam / 2 * regw
        return l_w
        
    #==============================================================================
    # main classification routine
    #==============================================================================
    def predict(self, test_matrix):
        test_size, w_size = test_matrix.shape
        predict_matrix = np.matrix(np.zeros((test_size, self.C)))
        w_x = test_matrix.dot(self.w)
        for c in range(0, self.C):
            predict_matrix[:, c] = np.exp(w_x[:, c]) / np.exp(w_x).sum(1)

        return np.array(predict_matrix.argmax(1)).T[0]
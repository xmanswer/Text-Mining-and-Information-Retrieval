# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:24:24 2016

@author: minxu

This is a class for PMF. contains necessary fields such as rating matrix, 
latent feature matrices, and learning parameters such as lambdas for U and V, 
learning rate, etc. It also contains necessary methods for constructing the 
latent feature matrices U and V, and predicted rating matrix R_hat, through 
gradient descent. It also provides the method for constructing training samples 
from rating matrix R.
"""

import scipy as sp
from scipy import sparse
import numpy as np
import math


class PMF:
    
    Iter = 100 #max number of iterations if not converge
    precision = 0.001 #lower bond of object function value for convergence
    lr = 0.001 #learning rate for gradient descent
    lambdaU = 0.0001 #regularization for U
    lambdaV = 0.0001 #regularization for U
    D = 5    
    
    def __init__(self, R, R_transpose, I, I_transpose, Iter=None, precision=None, lr=None, lambdaU=None, lambdaV=None, D=None):
        self.userSize, self.movieSize = R.shape
        self.R = R.todense()
        if Iter == None:
            Iter = self.Iter
        if precision == None:
            precision = self.precision
        if lr == None:
            lr = self.lr
        if lambdaU == None:
            lambdaU = self.lambdaU
        if lambdaV == None:
            lambdaV = self.lambdaV
        if D == None:
            D = self.D
        self.Rhat, self.U, self.V, self.error = self.gradient_desc(R, R_transpose, I, I_transpose)
        self.Rhat = self.Rhat + 3
        
    #==============================================================================
    # do gradient descent, in order to optimize latent feature matrices U, V
    # return the prediction score matrix Rhat which is the dot product of optimized
    # U and V, also return the error with iterations
    #==============================================================================
    def gradient_desc(self, R, R_transpose, I, I_transpose):
        #initialize randomly two latent feature matrices with correct score range
        
        U = (np.matrix(np.random.rand(self.D, self.userSize)) - 0.5)
        V = (np.matrix(np.random.rand(self.D, self.movieSize)) - 0.5)
        error = []
        for i in range(0, self.Iter):
            #print 'i: ' + str(i) 
            Unew = U - self.lr * self.obj_func_grad(U, V, R, I, self.lambdaU)
            Vnew = V - self.lr * self.obj_func_grad(V, U, R_transpose, I_transpose, self.lambdaV)
            U, V = Unew, Vnew
            Unew, Vnew = None, None
            E = self.obj_func(U, V, R, I)
            #print E
            error.append(E)
            if(E <= self.precision):
                break
        Rhat = U.transpose() * V
        return Rhat, U, V, error
    
    #==============================================================================
    # calculate gradient of the object function based on matrix U, V, R, I and lambda
    # fm1 is the matrix to be differentiated with
    #==============================================================================
    def obj_func_grad(self, fm1, fm2, R, I, lam):
        fm1_transpose = fm1.transpose() 
        temp = np.multiply((R - fm1_transpose * fm2), I.todense()) #user x movie
        grad = ((-2) * temp * fm2.transpose()) + 2 * lam * fm1_transpose
        fm1_transpose, temp = None, None
        return grad.transpose()
    
    #==============================================================================
    # object function, used to check convergence
    #==============================================================================
    def obj_func(self, U, V, R, I):
        temp = np.square(np.multiply((R - U.transpose() * V), I.todense()))
        sum1 = np.sum(temp)
        sum2 = np.sum(U.transpose() * U) * self.lambdaU
        sum3 = np.sum(V.transpose() * V) * self.lambdaV
        temp = None
        return  sum1 + sum2 + sum3
    
    #==============================================================================
    # construct the feature matrix (N (total samples) x D) for learning based on 
    # PMF matrix and return a N x 1 label vector
    #==============================================================================
    def feature_construction(self):
        features = []
        scores = []
        u_size = self.userSize
        cnt1 = 0
        cnt2 = 0
        
        for u in range(0, u_size):
            #print u
            nonzeros = self.R[u, :].nonzero()[1]
            for i in nonzeros:
                for j in nonzeros:
                    if i == j: 
                        continue 
                    diff = self.R[u, i] - self.R[u, j]
                    if math.fabs(diff) == 4:
                        v = np.multiply(self.U[:, u], self.V[:, i]) - np.multiply(self.U[:, u], self.V[:, j])
                        if diff > 0: y = 1
                        else: y = 0
                        features.append(np.array(v)[:, 0])
                        scores.append(y)
                        
                        if u == 1234:
                            cnt1 = cnt1 + 1
                        if u == 4321:
                            cnt2 = cnt2 + 1
                        
        print '# of training samples for 1234 is ' + str(cnt1)
        print '# of training samples for 4321 is ' + str(cnt2)
                        
        feature_matrix = np.matrix(features)
        label_vector = np.matrix(scores).transpose()
        return feature_matrix, label_vector
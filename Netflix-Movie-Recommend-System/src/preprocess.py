# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 21:51:33 2016

@author: minxu

This script contains functions used to parse traing set, development set 
and test set. It also contains functions to generate prediction results files
for development and test. 
"""

import scipy as sp
from scipy import sparse
import numpy as np

movieSize = 5392
userSize = 10916

#==============================================================================
# parse the trainning set .csv file, return sparse martices of userid as rows
# and movieid as columns 
#==============================================================================
def parseTrainingSet():
    fileName = 'train.csv'
    #main matrix where each element is a imputed score
    umMatrix = sp.sparse.lil_matrix((userSize, movieSize))
    #binary matrix where each element is a 1.0 if it is rated and 0.0 otherwise
    umMatrixBinary = sp.sparse.lil_matrix((userSize, movieSize))  
    
    with open(fileName) as f:
        for l in f:
            e = l.split(',')
            movieid = int(e[0])
            userid = int(e[1])
            score = float(e[2])                       
            imputedScore = score - 3.0
  
            #construct matrices
            umMatrix[userid, movieid] = imputedScore
            umMatrixBinary[userid, movieid] = 1.0

    #convert lil_matrix to csr_matrix
    umMatrix = umMatrix.tocsr()
    umMatrixBinary = umMatrixBinary.tocsr()
    return umMatrix, umMatrixBinary

#==============================================================================
# parse development/test set .csv file, return a dictionary where each key is 
# a movieid and each value is a list of userids belong to that movieid, return 
# a movieid list which keeps the order of the movieids
#==============================================================================
def parseDevTestSet(fileName):
    umdict = dict()
    userlist = []
    movielist = []
    with open(fileName) as f:
        for l in f:
            e = l.split(',')
            movieid = int(e[0])
            userid = int(e[1])
            if userid in umdict:
                umdict[userid].append(movieid)                
            else:
                umdict[userid] = [movieid]
            userlist.append(userid)
            movielist.append(movieid)
    
    return umdict, userlist, movielist

#==============================================================================
# write predicted class to file
#==============================================================================
def write_result(filename, res_score, userlist, movielist):
    with open(filename, 'w') as f:
        for i in range(0, len(userlist)):
            u, m = userlist[i], movielist[i]
            f.write(str(res_score[u][m]) + '\n')
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:24:24 2016
This is a program to generate predicted scores for user-movie queries
based on the training data set. It uses memory-based/model based collaborative
filtering with/without PCC, using either kNN algorithm or PMF for score predictions
Please refer to the README.txt file for more information
@author: minxu
"""

import scipy as sp
from scipy import sparse
import numpy as np
import matplotlib as mp
import math
import heapq
import time
import os
import subprocess

targetUID = 4321 #for corpus analysis
targetMID = 3 #for corpus analysis
movieSize = 5392
userSize = 10916
__sweepK__ = False #disable/enable k sweep in kNN
__sweepD__ = False #disable/enable D sweep in PMF
__generateTest__ = False #disable/enable generating test predictions

#==============================================================================
# parse the trainning set .csv file, return sparse martices of userid as rows
# and movieid as columns (except for umMatrixNormMovie and umMatrixPccMovie 
# which is the other way around), also generate some corpus statistics
#==============================================================================
def parseTrainingSet():
    fileName = 'train.csv'
    #main matrix where each element is a imputed score
    umMatrix = sp.sparse.lil_matrix((userSize, movieSize))
    #PCC matrix where each element is a centered normalized imputed score
    umMatrixPccMovie = sp.sparse.lil_matrix((movieSize, userSize))
    #binary matrix where each element is a 1.0 if it is rated and 0.0 otherwise
    umMatrixBinary = sp.sparse.lil_matrix((userSize, movieSize))
    
    #for convience of normalization
    rowSqSumList = [0] * userSize
    colSqSumList = [0] * movieSize
    movieSumList = [0] * movieSize
    movieRatedList = [0] * movieSize
    
    #corpus variables
    userSet = set()
    movieSet = set()    
    rate1cnt = 0
    rate3cnt = 0
    rate5cnt = 0
    average = 0
    totalRated = 0
    rate1user = 0
    rate3user = 0
    rate5user = 0
    averageTargetUser = 0
    totalRatedTargetUser = 0
    rate1movie = 0
    rate3movie = 0
    rate5movie = 0
    averageTargetMovie = 0
    totalRatedTargetMovie = 0
    
    #help function for increasing ratecnt and ratings for user/movie
    def incrementCnt(userid, movieid, ratecnt, rateuser, ratemovie):
        ratecnt = ratecnt + 1
        if userid == targetUID:
            rateuser = rateuser + 1
        if movieid == targetMID:
            ratemovie = ratemovie + 1
        return ratecnt, rateuser, ratemovie
    
    with open(fileName) as f:
        for l in f:
            e = l.split(',')
            movieid = int(e[0])
            userid = int(e[1])
            score = float(e[2])
            imputedScore = score - 3.0

            #for corpus stat
            userSet.add(userid)
            movieSet.add(movieid)   
            if score == 1: 
                rate1cnt, rate1user, rate1movie = incrementCnt(userid, movieid, rate1cnt, rate1user, rate1movie)
            if score == 3: 
                rate3cnt, rate3user, rate3movie = incrementCnt(userid, movieid, rate3cnt, rate3user, rate3movie)
            if score == 5: 
                rate5cnt, rate5user, rate5movie = incrementCnt(userid, movieid, rate5cnt, rate5user, rate5movie)
            
            average = average + score
            totalRated = totalRated + 1
            if userid == targetUID:
                averageTargetUser = averageTargetUser + score
                totalRatedTargetUser = totalRatedTargetUser + 1
            if movieid == targetMID:
                averageTargetMovie = averageTargetMovie + score
                totalRatedTargetMovie = totalRatedTargetMovie + 1
                
            #construct matrices
            #notice that umMatrixPccMovie needs non imputed score to start with
            umMatrix[userid, movieid] = imputedScore
            umMatrixPccMovie[movieid, userid] = score
            umMatrixBinary[userid, movieid] = 1.0
            #accumalte the original score for each movie for average calculation
            movieSumList[movieid] = movieSumList[movieid] + score
            movieRatedList[movieid] = movieRatedList[movieid] + 1
            #accumulate the sum of each row/col for normalization
            rowSqSumList[userid] = rowSqSumList[userid] + imputedScore * imputedScore
            colSqSumList[movieid] = colSqSumList[movieid] + imputedScore * imputedScore

    #umMatrixNormUser is user x movie
    #umMatrixNormMovie and umMatrixPccMovie is movie x user
    #each row of them is normalized by the mod of row
    umMatrixNormUser = sp.sparse.lil_matrix((userSize, movieSize))
    umMatrixNormMovie = sp.sparse.lil_matrix((movieSize, userSize))
    umMatrix_transpose = umMatrix.transpose()
    
    #normalize row based on the mod of each row
    for i in range(0, userSize):
        if(rowSqSumList[i] == 0): continue
        rowSqSum = math.sqrt(rowSqSumList[i])
        umMatrixNormUser[i,] = umMatrix[i,] / rowSqSum
        
    for i in range(0, movieSize):
        #normalize col based on the mod of col
        if(colSqSumList[i] == 0): continue
        colSqSum = math.sqrt(colSqSumList[i])
        umMatrixNormMovie[i,] = umMatrix_transpose[i,] / colSqSum
        
        #do standarization on each col
        mean = movieSumList[i] / movieRatedList[movieid]
        nonzeros = umMatrixPccMovie[i,].nonzero()[1] #get non zero indexes
        for index in nonzeros: #center non zero elements and do imputation
            umMatrixPccMovie[i,index] = umMatrixPccMovie[i,index] - mean - 3
        
        #then do nomarlization based on imputed centered matrix
        norm = math.sqrt((umMatrixPccMovie[i,].dot(umMatrixPccMovie[i,].transpose()))[0,0])
        if(norm == 0): continue
        umMatrixPccMovie[i,] = umMatrixPccMovie[i,] / norm

    #write corpus stat
    average = average / totalRated  
    averageTargetUser = averageTargetUser / totalRatedTargetUser
    averageTargetMovie = averageTargetMovie / totalRatedTargetMovie
    
    corpStatFile.write('total number of movies is: ' + str(len(movieSet)) + '\n')
    corpStatFile.write('total number of users is: ' + str(len(userSet)) + '\n')
    corpStatFile.write('the number of times any movie was rated 1: ' + str(rate1cnt) + '\n')
    corpStatFile.write('the number of times any movie was rated 3: ' + str(rate3cnt) + '\n')
    corpStatFile.write('the number of times any movie was rated 5: ' + str(rate5cnt) + '\n')
    corpStatFile.write('the average rating across all: ' + str(average) + '\n')
    corpStatFile.write('the number of movies rated: ' + str(totalRatedTargetUser) + '\n')
    corpStatFile.write('the number of times the user rate 1: ' + str(rate1user) + '\n')
    corpStatFile.write('the number of times the user rate 3: ' + str(rate3user) + '\n')
    corpStatFile.write('the number of times the user rate 5: ' + str(rate5user) + '\n')
    corpStatFile.write('the average rating of this user: ' + str(averageTargetUser) + '\n')
    corpStatFile.write('the number of users rated this movie: ' + str(totalRatedTargetMovie) + '\n')
    corpStatFile.write('the number of times the movie rated 1: ' + str(rate1movie) + '\n')
    corpStatFile.write('the number of times the movie rated 3: ' + str(rate3movie) + '\n')
    corpStatFile.write('the number of times the movie rated 5: ' + str(rate5movie) + '\n')
    corpStatFile.write('the average rating of this movie: ' + str(averageTargetMovie) + '\n')
        
    #convert lil_matrix to csr_matrix
    umMatrix = umMatrix.tocsr()
    umMatrixNormUser = umMatrixNormUser.tocsr()
    umMatrixNormMovie = umMatrixNormMovie.tocsr()
    umMatrixPccMovie = umMatrixPccMovie.tocsr()
    umMatrixBinary = umMatrixBinary.tocsr()
    
    return {'umMatrix':umMatrix, 'umMatrixNormUser':umMatrixNormUser, 
    'umMatrixNormMovie':umMatrixNormMovie, 'umMatrixPccMovie':umMatrixPccMovie,
    'umMatrixBinary':umMatrixBinary}

#==============================================================================
# parse development/test set .csv file, return a dictionary where each key is 
# a movieid and each value is a list of userids belong to that movieid, return 
# a movieid list which keeps the order of the movieids
#==============================================================================
def parseDevTestSet(fileName):
    umdict = dict()
    movieidList = []
    with open(fileName) as f:
        for l in f:
            e = l.split(',')
            movieid = int(e[0])
            userid = int(e[1])
            if movieid in umdict:
                umdict[movieid].append(userid)
            else:
                umdict[movieid] = [userid]
                movieidList.append(movieid)
    
    return umdict, movieidList

#==============================================================================
# do kNN for given id, return top k similarity id list and id:sim dictionary
#==============================================================================
def kNN(k, um, qm, expScoreFunc):
    simMatrix = qm.dot(um.transpose()) #similarity matrix
    Average = []
    Weighted = []

    for movieid in movieidList:
        userList = umdict[movieid]
        for uid in userList:
            if expScoreFunc == exp1Score:
                #get scores for memory-based model using dot product sim
                simDict = dict(enumerate((simMatrix[uid,].toarray())[0].tolist()))
            else:
                simDict = dict(enumerate((simMatrix[movieid,].toarray())[0].tolist()))
            score, weightedScore = expScoreFunc(uid, movieid, simMatrix, k, simDict)
            Average.append(score)
            Weighted.append(weightedScore)
            
    simMatrix = None
    return Average, Weighted    
            
#==============================================================================
# exp1: calculate the average and weighted score based on kNN users
#==============================================================================
def exp1Score(userid, movieid, A, k, userSimDict):
    topkUserList = heapq.nlargest(k + 1, userSimDict, key = userSimDict.get)
    score = 0
    totalWeight = 0
    
    #get the average score
    for u in topkUserList:
        if(u == userid): continue
        score = score + 1.0 / k * umMatrix[u, movieid]
        totalWeight = totalWeight + math.fabs(userSimDict[u])
    
    #get the weighted score
    weightedScore = 0
    if totalWeight != 0:
        for u in topkUserList:
            if(u == userid): continue
            weightedScore = weightedScore + userSimDict[u] / totalWeight * umMatrix[u, movieid]
    else: #no totalWeight means that this user is empty
        score = umMatrix[:,movieid].sum() / (userSize-1)
        weightedScore = score
        
    score = score + 3.0
    weightedScore = weightedScore + 3.0
    return score, weightedScore

#==============================================================================
# exp1: calculate the average and weighted score based on kNN movies and matrix A
#==============================================================================
def exp2Score(userid, movieid, A, k, movieSimDict):
    topkMovieList = heapq.nlargest(k + 1, movieSimDict, key = movieSimDict.get)
    score = 0
    totalWeight = 0

    #get the average score
    for m in topkMovieList:
        if(m == movieid): continue
        score = score + 1.0 / k * umMatrix[userid, m]
        totalWeight = totalWeight + math.fabs(movieSimDict[m])
    
    #get the weighted score
    weightedScore = 0
    if totalWeight != 0:
        for m in topkMovieList:
            if(m == movieid): continue
            weightedScore = weightedScore + movieSimDict[m] / totalWeight * umMatrix[userid, m]
    else: #no totalWeight means that this movie is empty
        score = umMatrix[userid,].sum() / (movieSize-1)
        weightedScore = score
        
    score = score + 3.0
    weightedScore = weightedScore + 3.0
    return score, weightedScore

#==============================================================================
# generate score files based on given experiment
#==============================================================================
def generateScore(expStr, dotAverage, dotWeighted, cosAverage, cosWeighted):
    print 'start generating scores for ' + expStr
    
    f1 = open(expStr + '_average_cosine', 'w')
    f2 = open(expStr + '_weighted_cosine', 'w')
    f3 = open(expStr + '_average_dot', 'w')
    f4 = open(expStr + '_weighted_dot', 'w')  
    
    for i in range(0, len(dotAverage)):
        f1.write(str(dotAverage[i]) + '\n')
        f2.write(str(dotWeighted[i]) + '\n')
        f3.write(str(cosAverage[i]) + '\n')
        f4.write(str(cosWeighted[i]) + '\n')
        
    f1.close()
    f2.close()
    f3.close()
    f4.close()

#==============================================================================
# used to get corpus stat on given user and movie for their kNN top 5
#==============================================================================
def corpusKNN(umMatrix, umMatrixNormUser, umMatrixNormMovie):
    #help function for corpusKNN
    def corpusKNNhelp(simMatrix):
        simDict = dict(enumerate((simMatrix.toarray())[0].tolist()))
        top5 = heapq.nlargest(6, simDict, key = simDict.get)
        for i in top5[1:len(top5)]:
            corpStatFile.write(str(i) + ':' + str(simDict[i]) + ' ')
        corpStatFile.write('\n')
        simMatrix, simDict = None, None
    
    #analyze user 4321 kNN top 5 from dot product similarity
    simMatrix = umMatrix[targetUID,].dot(umMatrix.transpose())
    corpStatFile.write('Top 5 NNs of user 4321 using dot product similarity: ')
    corpusKNNhelp(simMatrix)
    
    #analyze user 4321 kNN top 5 from cosine similarity
    simMatrix = umMatrixNormUser[targetUID,].dot(umMatrixNormUser.transpose())
    corpStatFile.write('Top 5 NNs of user 4321 using cosine similarity: ')
    corpusKNNhelp(simMatrix)
    
    #analyze user 3 kNN top 5 from dot product similarity
    simMatrix = umMatrix[:,targetMID].transpose().dot(umMatrix)
    corpStatFile.write('Top 5 NNs of movie 3 using dot product similarity: ')
    corpusKNNhelp(simMatrix)
    
    #analyze user 3 kNN top 5 from cosine similarity
    simMatrix = umMatrixNormMovie[targetMID,].dot(umMatrixNormMovie.transpose())
    corpStatFile.write('Top 5 NNs of movie 3 using cosine similarity: ')
    corpusKNNhelp(simMatrix)

#==============================================================================
# do gradient descent, in order to optimize latent feature matrices U, V
# return the prediction score matrix Rhat which is the dot product of optimized
# U and V, also return the error with iterations
#==============================================================================
def gradientDesc(Iter, precision, lr, R, R_transpose, I, I_transpose, lambdaU, lambdaV, D):
    #initialize randomly two latent feature matrices with correct score range
    U = (np.matrix(np.random.rand(D, userSize)) - 0.5)
    V = (np.matrix(np.random.rand(D, movieSize)) - 0.5)
    error = []
    for i in range(0, Iter):
        #print 'i: ' + str(i) 
        Unew = U - lr * objFuncGrad(U, V, R, I, lambdaU, D)
        Vnew = V - lr * objFuncGrad(V, U, R_transpose, I_transpose, lambdaV, D)
        U, V = Unew, Vnew
        Unew, Vnew = None, None
        E = objFunc(U, V, R, I, lambdaU, lambdaV)
        #print E
        error.append(E)
        if(E <= precision):
            break
    Rhat = U.transpose() * V
    return Rhat, error

#==============================================================================
# calculate gradient of the object function based on matrix U, V, R, I and lambda
# fm1 is the matrix to be differentiated with
#==============================================================================
def objFuncGrad(fm1, fm2, R, I, Lambda, D):
    fm1_transpose = fm1.transpose() 
    temp = np.multiply((R - fm1_transpose * fm2), I.todense()) #user x movie
    grad = ((-2) * temp * fm2.transpose()) + 2 * Lambda * fm1_transpose
    fm1_transpose, temp = None, None
    return grad.transpose()

#==============================================================================
# object function, used to check convergence
#==============================================================================
def objFunc(U, V, R, I, lambdaU, lambdaV):
    temp = np.square(np.multiply((R - U.transpose() * V), I.todense()))
    sum1 = np.sum(temp)
    sum2 = np.sum(U.transpose() * U) * lambdaU
    sum3 = np.sum(V.transpose() * V) * lambdaV
    temp = None
    return  sum1 + sum2 + sum3

#==============================================================================
# generate scores for PMF method
#==============================================================================
def generateScorePMF(Rhat, midl, umd, fileName):
    print 'start generating scores for PMF'
    
    f = open(fileName, 'w')
    for movieid in midl:
        userList = umd[movieid]
        for uid in userList:
            score = Rhat[uid, movieid]
            f.write(str(score) + '\n')
    
    f.close()
    
#==============================================================================
# main script
#==============================================================================

#==============================================================================
# parsing all files
#==============================================================================
corpStatFile = open('corpstat.txt', 'w')
trainingSetResult = parseTrainingSet()
umMatrix = trainingSetResult['umMatrix']
umMatrix_transpose = umMatrix.transpose()
umMatrixNormUser = trainingSetResult['umMatrixNormUser']
umMatrixNormMovie = trainingSetResult['umMatrixNormMovie']
umMatrixPccMovie = trainingSetResult['umMatrixPccMovie']
umMatrixBinary = trainingSetResult['umMatrixBinary']
umdict, movieidList = parseDevTestSet('dev.csv')

#==============================================================================
# finish corpus stat analysis first
#==============================================================================
corpusKNN(umMatrix, umMatrixNormUser, umMatrixNormMovie)
corpStatFile.close()

#==============================================================================
# development for experiment 1 - 3, find the best k and CF methods
#==============================================================================
#set list of k values, disable the sweep through __sweepK__
if __sweepK__:
    K = [10, 100, 500]
else:
    K = [10]

#do development, may loop different k
for k in K:
    print '....start exp1, exp2 and exp3 based on k = ' + str(k) + '....'
    #generate scores for memory-based CF using user-user similarity
    t = time.time()
    dotAverage, dotWeighted = kNN(k, umMatrix, umMatrix, exp1Score)
    print 'user-user dot sim eclipse time: ' + str(time.time() - t)
    
    t = time.time()
    cosAverage, cosWeighted = kNN(k, umMatrixNormUser, umMatrixNormUser, exp1Score)
    print 'user-user cosine sim eclipse time: ' + str(time.time() - t)
    
    generateScore('exp1', dotAverage, dotWeighted, cosAverage, cosWeighted)
    
    #generate scores for model-based CF using movie-movie similarity
    t = time.time()
    dotAverage, dotWeighted = kNN(k, umMatrix_transpose, umMatrix_transpose, exp2Score)
    print 'movie-movie dot sim eclipse time: ' + str(time.time() - t)
    
    t = time.time()
    cosAverage, cosWeighted = kNN(k, umMatrixNormMovie, umMatrixNormMovie, exp2Score)
    print 'movie-movie cosine sim eclipse time: ' + str(time.time() - t)
    
    generateScore('exp2', dotAverage, dotWeighted, cosAverage, cosWeighted)
    
    #generate scores for model-based CF with PCC standarization
    t = time.time()
    dotAverage, dotWeighted = kNN(k, umMatrixPccMovie, umMatrixPccMovie, exp2Score)
    print 'movie-rating standarized eclipse time: ' + str(time.time() - t)
    
    generateScore('exp3', dotAverage, dotWeighted, dotAverage, dotWeighted)
    cdir = os.getcwd()
    
    for f in os.listdir(cdir):
        if f.startswith('exp'):
            stdout = subprocess.check_output("python eval_rmse.py dev.golden " + f)
            print f + ':\t' + stdout

#==============================================================================
# do development for PMF, use parameters for best RMSE
#==============================================================================
Iter = 100 #max number of iterations if not converge
precision = 0.001 #lower bond of object function value for convergence
lr = 0.001 #learning rate for gradient descent
lambdaU = 0.0001 #regularization for U
lambdaV = 0.0001 #regularization for U
R = umMatrix  #rating matrix
I = umMatrixBinary #binary rating matrix where the element is 1 if rated
R_transpose = umMatrix_transpose
I_transpose = I.transpose()

#set the list of latent feature size, can disable sweep by setting __sweepD__
if __sweepD__:
    D = [2, 5, 10, 20, 50]
else:
    D = [5]

for d in D:
    t = time.time()
    Rhat, error = gradientDesc(Iter, precision, lr, R, R_transpose, I, I_transpose, lambdaU, lambdaV, d)
    print 'PMF eclipse time for D = ' + str(d) + ': ' + str(time.time() - t)
    f = 'PMF_score_' + str(d)
    Rhat = Rhat + 3
    generateScorePMF(Rhat, movieidList, umdict, f)
    print "PMF_score for D = ' + str(d) + ':\t" + subprocess.check_output("python eval_rmse.py dev.golden " + f)

#==============================================================================
# generate output files for testset
#==============================================================================
if __generateTest__:
    umdictTest, movieidListTest = parseDevTestSet('test.csv')
    D = 5 #the best D value
    Rhat, error = gradientDesc(Iter, precision, lr, R, R_transpose, I, I_transpose, lambdaU, lambdaV, D)
    generateScorePMF(Rhat, movieidListTest, umdictTest, 'predictions.txt')
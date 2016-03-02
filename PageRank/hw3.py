# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:41:23 2016

@author: Min Xu, andrewID:minxu

Locate this program in the same folder of files such as doc_topics.txt
Also locate the indri-lists folder in the same folder of this program
This program will parse all txt files and generate neccessary data structures
It will perform NS, WS and CS on GPR, QTSPR and PTSPR, generating nine
different output files
It will also generate three files for submission

"""

import scipy as sp
from scipy import sparse
import numpy as np
import math
import os
import time

doclen = 81433
queryLen = 6
userLen = 20
clusters = 12
damping = 0.8 #1 - damping factor
Iter = 10 #iteration number
weightSum = 3
weightCustom = 0.99
BetaQuery = 0.99 #actually real beta should be (1-damping) * BetaQuery
BetaUser= 0.99

#==============================================================================
# parse transition matrix, return trans matrix with size 81433 x 81433
# return p0 matrix with size 81433 x 1
# use lil_matrix for high speed insert
# notice that id is 1-based in txt, but 0-based in matrix
#==============================================================================
def parseTransition(fileName):
    trans = sp.sparse.lil_matrix((doclen, doclen))
    p0 = sp.sparse.lil_matrix((doclen, 1))
    
    #first go through and count the outlinks for each docid
    #initial value will be at least 1 outlink (itself)
    hm = [1 for i in range(doclen)]
    with open(fileName) as f:        
        for l0 in f:
            nums = l0.split(' ')
            docid = int(nums[0])-1
            hm[docid] = hm[docid] + 1
    
    #go through again to calculate transition value for each pair
    with open(fileName) as f:
        for l in f:
            nums = l.split(' ')
            docid_out = int(nums[0])-1
            docid_in = int(nums[1])-1
            
            trans[docid_out, docid_in] = 1.0 / (hm[docid_out])
            trans[docid_out, docid_out] = 1.0 / (hm[docid_out])
            trans[docid_in, docid_in] = 1.0 / (hm[docid_in])
            p0[docid_out, 0] = 1.0/float(doclen)
            p0[docid_in, 0] = 1.0/float(doclen)
            
    trans = trans.tocsr()
    p0 = p0.tocsr()
    return {'trans':trans, 'p0':p0}

#==============================================================================
# parse distribution, return a dict of user-query : topic distributions list
#==============================================================================
def parseDistr(fileName):
    distr = dict()
    with open(fileName) as f:
        for l in f:
            nums = l.split(' ')
            probList = []
            for pair in nums[2:len(nums)]:
                pair = pair.split(':')
                probList.append(float(pair[1]))
            keystring = nums[0] + '-' + nums[1]
            distr[keystring] = probList
    return distr
#==============================================================================
# Parse relevance file, return a dict of queryname : querydict pairs
# where each querydict is a dict of docids and scores
#==============================================================================
def parseRelevance(folderName):
    queryDict = dict()
    for fileName in os.listdir(folderName):
        relevScore = dict()
        queryName = fileName.split('.')[0]
        with open(folderName + '/' + fileName) as f:
            for l in f:
                nums = l.split(' ')
                docid = int(nums[2])-1
                score = float(nums[4])
                relevScore[docid] = score
        queryDict[queryName] = relevScore
    return queryDict

#==============================================================================
# parse doc_topics file a return a sparse matrix of size 81433 x 12
# where it is non-zero if the docid belongs to the topic, value is 
# normalized based on the sum of the col so each col of this matrix is a pt
#==============================================================================
def parseDocTopic(fileName):
    docTopicMatrix = sp.sparse.lil_matrix((doclen, clusters))
    topicSum = [0] * clusters
    with open(fileName) as f:
        for l in f:
            nums = l.split(' ')
            topicid = int(nums[1]) - 1
            docid = int(nums[0]) - 1
            topicSum[topicid] = topicSum[topicid] + 1
            docTopicMatrix[docid, topicid] = 1

    for i in range(clusters):
        docTopicMatrix[:, i] = docTopicMatrix[:, i] / topicSum[i]
    
    return docTopicMatrix.tocsr()

#==============================================================================
# do PageRank iteration based on different inputs, return 81433 x 1 scores matrix
#==============================================================================
def pageRankIteration(pgMatrix, pgParam):
    r = pgMatrix['r']
    trans_transpose = pgMatrix['trans_transpose']
    p0 = pgMatrix['p0']
    pt = pgMatrix['pt']
    damping = pgParam['damping']
    beta = pgParam['beta']
    Iter = pgParam['Iter']
        
    for i in range(Iter):
        r = damping * trans_transpose.dot(r) + (1 - damping) * (1 - beta) * p0 + (1 - damping) * beta * pt
    return r

#==============================================================================
# combine the PageRank score and the relevance score for each
# relevant document in the query, output to the file for all queries
# weight will be used to adjust the weights between PageRank and relevance
#==============================================================================
def combineRelevance(uqr, queryDict, weight, fileName, combineFunc, getRFunc):
    with open(fileName, 'w') as f:
        for queryName in queryDict:
            r = getRFunc(uqr, queryName)
            scores = queryDict[queryName]
            newscores = dict()
            for docid in scores:
                pgscore = r[docid,0]
                rvscore = np.exp(scores[docid])
                newscores[docid] = combineFunc(weight, pgscore, rvscore)
            i = 1
            for k in sorted(newscores, key = newscores.get, reverse=True):
                f.write(queryName + ' Q0 ' + str(k+1) + ' ' + str(i) + ' ' + str(newscores[k]) + ' run\n')
                i = i + 1

#if uqr is a dict of queryName : rank matrix pairs, get the score matrix
def getRforQueryName(uqr, queryName):
    return uqr[queryName]

#if just general PR, just return the r since it is a score matrix
def getRforGeneral(uqr, queryName):
    return uqr
#==============================================================================
# functions for combining PageRank score and relevance score
#==============================================================================
#calculate score for no search case
def noSearch(weight, pgscore, rvscore):
    return np.log(pgscore)

#calculate score for weighted sum
def weightedSum(weight, pgscore, rvscore):
    return np.log((1- weight) * pgscore + weight * rvscore)

#calculate score for custom method
def customSum(weight, pgscore, rvscore):
    pgscore = np.log(pgscore)
    rvscore = np.log(rvscore)
    return 1.0 / ((1 - weight) / pgscore + weight / rvscore)

#==============================================================================
# return rank matrix for different topics based on query-topic distributions
#==============================================================================
def tsPageRank(docTopicMatrix, probList, pgMatrix, pgParam):
    pgScoreMatrix = sp.sparse.lil_matrix((doclen, 1))    
    for i in range(clusters):
        pgMatrix['pt'] = docTopicMatrix[:, i]
        pgScoreMatrix = pgScoreMatrix + probList[i] * pageRankIteration(pgMatrix, pgParam)
    
    return pgScoreMatrix

#==============================================================================
# for each user-query pair, return the final rank based on combined scores
# of topic sensitive PR scores and relevance search scores
# uqDict can be either query sensitive (qd) or user sensitive (ud) 
# each user-query pair has a corresponding rank score matrix
#==============================================================================
def topicSensRank(uqDict, pgMatrix, pgParam, docTopicMatrix):
    uqr = dict()
    for uqPair in uqDict:
        probList = uqDict[uqPair]
        uqr[uqPair] = tsPageRank(docTopicMatrix, probList, pgMatrix, pgParam)
    
    return uqr

#==============================================================================
# this is used to output required files
#==============================================================================
def outPutGPR(uqr, fileName, queryName, getRFunc):
    r = getRFunc(uqr, queryName)
    with open(fileName, 'w') as f:
        pgdict = dict()
        for i in range(doclen):
            pgdict[i] = r[i, 0]
        for k in sorted(pgdict, key = pgdict.get, reverse=True):
            f.write(str(k+1) + ' ' + str(pgdict[k]) + '\n')

#==============================================================================
# main script part    
#==============================================================================
tp = parseTransition('transition.txt')
trans = tp['trans']
trans_transpose = trans.transpose()
p0 = tp['p0']
qd = parseDistr('query-topic-distro.txt')
ud = parseDistr('user-topic-distro.txt')
queryDict = parseRelevance('indri-lists')
docTopicMatrix = parseDocTopic('doc_topics.txt')

#specify parameters and matrices for PR
pgMatrix = dict()
pgParam = dict()
pgMatrix['r'] = p0 #initial r is just p0
pgMatrix['trans_transpose'] = trans_transpose
pgMatrix['p0'] = p0
pgMatrix['pt'] = 0
pgParam['damping'] = damping
pgParam['beta'] = 0
pgParam['Iter'] = Iter


#Global PageRank for NS, WS and CS

r = pageRankIteration(pgMatrix, pgParam) #global PageRank score matrix
outPutGPR(r, 'GPR-10.txt', None, getRforGeneral)

currTime = time.time()

combineRelevance(r, queryDict, None, 'gpr_ns.txt', noSearch, getRforGeneral)
currTime = time.time() - currTime
print 'start GPR NS, current time is ' + str(currTime)

combineRelevance(r, queryDict, weightSum, 'gpr_ws.txt', weightedSum, getRforGeneral)
currTime = time.time() - currTime
print 'start GPR WS, current time is ' + str(currTime)

combineRelevance(r, queryDict, weightCustom, 'gpr_cs.txt', customSum, getRforGeneral)
currTime = time.time() - currTime
print 'start GPR CS, current time is ' + str(currTime)


#Query-sensitive PageRank for NS, WS and CS
pgMatrix['r'] = p0
pgParam['beta'] = BetaQuery
uqr = topicSensRank(qd, pgMatrix, pgParam, docTopicMatrix)
outPutGPR(uqr, 'QTSPPR-U2Q1-10.txt', '2-1', getRforQueryName)

currTime = time.time()

combineRelevance(uqr, queryDict, None, 'qtspr_ns.txt', noSearch, getRforQueryName)
currTime = time.time() - currTime
print 'start QTGPR NS, current time is ' + str(currTime)

combineRelevance(uqr, queryDict, weightSum, 'qtspr_ws.txt', weightedSum, getRforQueryName)
currTime = time.time() - currTime
print 'start QTGPR WS, current time is ' + str(currTime)

combineRelevance(uqr, queryDict, weightCustom, 'qtspr_cs.txt', customSum, getRforQueryName)
currTime = time.time() - currTime
print 'start QTGPR CS, current time is ' + str(currTime)


#User-sensitive PageRank for NS, WS and CS
pgMatrix['r'] = p0
pgParam['beta'] = BetaUser
uqr = topicSensRank(ud, pgMatrix, pgParam, docTopicMatrix)
outPutGPR(uqr, 'PTSPPR-U2Q1-10.txt', '2-1', getRforQueryName)

currTime = time.time()

combineRelevance(uqr, queryDict, None, 'ptspr_ns.txt', noSearch, getRforQueryName)
currTime = time.time() - currTime
print 'start PTGPR NS, current time is ' + str(currTime)

combineRelevance(uqr, queryDict, weightSum, 'ptspr_ws.txt', weightedSum, getRforQueryName)
currTime = time.time() - currTime
print 'start PTGPR WS, current time is ' + str(currTime)

combineRelevance(uqr, queryDict, weightCustom, 'ptspr_cs.txt', customSum, getRforQueryName)
currTime = time.time() - currTime
print 'start PTGPR CS, current time is ' + str(currTime)


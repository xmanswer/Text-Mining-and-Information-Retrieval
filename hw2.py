# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:41:23 2016

@author: Min Xu, andrewID:minxu

This is a script for processing data read from *.docVectors files
Please put this script in the same folder of all *.docVectors filse
The evaluation file eval.py and *.gold_standards also has to be 
in this folder

To run the script, just do python hw2.py. No additional args needed

It will generate four docCluster files corresponding to three experiments, 
which have format of "docID clusterID"

"""

import subprocess
import scipy as sp
from scipy import sparse
import numpy as np
import matplotlib as mp
import math
import random
import bisect

#==============================================================================
# do corpus stat analysis on given file, return three vectors
#==============================================================================
def corpusStat(fileType):
    doclist = [] #store term : freq map for each doc, asc in docId
    termdict = dict() #store term : total freq for each term, asc in termId
    termdf = dict() #store term : doc list for each term, asc in termId
    
    with open(fileType + '.docVectors') as dv:
        wordSize, docId = 0, 0
        for l in dv:
            termpairs = l.split()
            termmap = dict()
            for tp in termpairs:
                tp = tp.split(':')
                term = int(tp[0])
                freq = int(tp[1])
                termmap[term] = freq
                wordSize = wordSize + freq
                
                #update term : total freq
                if term in termdict:
                    termdict[term] = termdict[term] + freq
                else:
                    termdict[term] = freq
                
                #update term : doc list
                if term in termdf:
                    termdf[term].add(docId)
                else:
                    docSet = set()
                    docSet.add(docId)
                    termdf[term] = docSet
                    
            #update doclist for term : freq pairs
            doclist.append(termmap)
            docId = docId + 1
    
    #used to answer the last question for corpus exploration
    #find all of the word ids that occurred exactly twice in the first doc
    ids = dict()
    for k, v in doclist[0].iteritems():
        if v == 2:
            ids[k] = math.log(float(len(doclist)) / float(termdict[k]))
    
    sparseDocList = []
    sparseDoctListTfIDf = []
    #create sparse vectors for each doc in doclist
    for doc in doclist:
        sv = sp.sparse.csc_matrix((len(termdict), 1))
        sv_tfidf = sp.sparse.csc_matrix((len(termdict), 1))
        for k, v in doc.iteritems():
            tf = float(v) #term frequency in this document
            sv[k,0] = tf
            #inverse document frequency
            idf = math.log(float(len(doclist)) / float(termdict[k]) + 1) 
            sv_tfidf[k,0] = tf * idf
            
        sparseDocList.append(sv)
        sparseDoctListTfIDf.append(sv_tfidf)
    
    print '**************** corpus analysis start ****************'
    print 'corpus stat for the ' + fileType
    print 'total number of documents is ' + str(len(doclist))
    print 'total number of words is ' + str(wordSize)
    print 'total number of unique words is ' + str(len(termdict))
    print 'average number of unique words per document is ' \
    + str(len(termdict) / len(doclist))  
    print 'total number of unique words for the first doc is ' \
    + str(len(doclist[0]))
    print 'all of word ids occurred twice in the first doc are: '
    print ids
    print '**************** corpus analysis end ****************'
    
    return {'doclist':sparseDocList, 'doclist_tfIdf': sparseDoctListTfIDf, 'termdict':termdict, 'termdf':termdf}

#==============================================================================
# compute the cosine similarity between two documents
#==============================================================================
def sim(docVect1, docVect2):
    result = docVect1.transpose().dot(docVect2).sum()
    dis1 = math.sqrt(docVect1.transpose().dot(docVect1).sum())
    dis2 = math.sqrt(docVect2.transpose().dot(docVect2).sum())
    return result / (dis1 * dis2)       


#==============================================================================
# compute the Euclidean distance between two documents    
#==============================================================================
def distance(doclist1, doclist2):
    diff = doclist1 - doclist2
    return math.sqrt(diff.transpose().dot(diff).sum())


def euclideanSim(doclist1, doclist2):
    dist = distance(doclist1, doclist2)
    if dist == 0:
        return float("inf")
    else:
        return 1 / dist
#==============================================================================
# do one iteration of kmeans
# return cluster id for each doc and the updated centroids
#==============================================================================
def kmeans(doclist, centroids, smallChange, oldClass, termdictSize, simFunc):
    docCluster = dict() #store docId : clusterId
    doclistSize = len(doclist)
    #store [clusterId] = assigned doc number
    centroidCounter = [0] * len(centroids)
    newCentroids = [sp.sparse.csc_matrix((termdictSize, 1))] * len(centroids)
    mismatch = 0
    
    for i in range(0, doclistSize): #for each document
        maxScore = float("-inf") #initial score is -inf 
        for c in range(0, len(centroids)): #for each cluster compute the similarity score
            score = simFunc(doclist[i], centroids[c])
            if score > maxScore: #assign if higher than current max
                maxScore, assignedClass = score, c
        docCluster[i] = assignedClass
        if oldClass[i] is not docCluster[i]: #check cluster change
            mismatch = mismatch + 1
        
        #accumulate number of docs assigned to each centroid for averaging
        centroidCounter[assignedClass] = centroidCounter[assignedClass]+1
        #add current doc to new centroid
        newCentroids[assignedClass] = newCentroids[assignedClass] + doclist[i]
    
    #see if total mismatched cluster number is smaller than threshold
    converge = False
    if mismatch <= smallChange:
        converge = True
        
    for c in range(0, len(centroids)):
        #if no doc assigned to this cluster, random generate it
        if centroidCounter[c] == 0:
            newCentroids[c] = sp.sparse.rand(termdictSize, 1, 0.01, 'csc')
        else:
            newCentroids[c] = newCentroids[c] / float(centroidCounter[c])
            
    return {'docCluster':docCluster, 'centroids':newCentroids, 'converge' : converge}
    

#==============================================================================
# randomly sample seedNumber seed centroids
#==============================================================================
def randomSeeds(seedNumber, doclist):
    return [doclist[c] for c in random.sample(
    [i for i in range(0, len(doclist))], seedNumber)]
    
#==============================================================================
# select seed centroids using kmeans++
#==============================================================================
def kmeansPlusPlus(seedNumber, doclist):
    #pick the first center at uniformly random
    centroids = []
    centroids.append(random.choice(doclist))

    for c in range(1, seedNumber):
        total = 0
        cumWeights = []
        for i in range(0, len(doclist)):
            weight = math.pow((distance(doclist[i], centroids[c-1])), 2)
            total = total + weight
            cumWeights.append(total)
        
        newseed = bisect.bisect_right(cumWeights, random.random() * total)
        centroids.append(doclist[newseed])
        
    return centroids

#==============================================================================
# start clustering for given inputs parameters
#==============================================================================
def runClustering(C, smallChange, filePath, devdata, initi, simFunc, useTfIdf = False):
    if useTfIdf:
        doclist = devdata['doclist_tfIdf']
    else:
        doclist = devdata['doclist']
    termdictSize = len(devdata['termdict'])
    
    maxIter = 10 #max iteration
    Iter = 0
    
    #initialize centroids
    centroids = initi(C, doclist)
    
    docCluster = {k:-1 for k in range(len(doclist))}
    converge = False
    
    #do kmeans until converge or reach max iterations
    while not converge and Iter < maxIter:
        kmeansRes = kmeans(doclist, centroids, smallChange, 
                           docCluster, termdictSize, simFunc)
        centroids = kmeansRes['centroids']
        docCluster = kmeansRes['docCluster']
        converge = kmeansRes['converge']
        Iter = Iter + 1
    
    with open(filePath, 'w') as f:
        for k, v in docCluster.iteritems():
            f.write(str(k) + ' ' + str(v) + '\n')
            
#==============================================================================
# main script for experiment
#==============================================================================
devdata = corpusStat('HW2_dev')

#cluster numbers
Clist = [250]
#thresholds of changed cluster for convergence
cnvgList = [10]

#==============================================================================
# random initilization, cosine similarity
#==============================================================================
print '.....do kmeans for random initilization.....'
for i in range(0, len(Clist)):
    for j in range(0, len(cnvgList)):
        filePath = 'docCluster_randInit_' + str(Clist[i]) + '_cnvg_' + str(cnvgList[j]) + '.txt'
        runClustering(Clist[i], cnvgList[j], filePath, 
                      devdata, randomSeeds, sim)

#==============================================================================
# kmeans++ initilization, cosine similarity
#==============================================================================
print '.....do kmeans for kmeans++ initilization.....'
for i in range(0, len(Clist)):
    for j in range(0, len(cnvgList)):
        filePath = 'docCluster_kmppInit_' + str(Clist[i]) + '_cnvg_' + str(cnvgList[j]) + '.txt'
        runClustering(Clist[i], cnvgList[j], filePath, 
                      devdata, kmeansPlusPlus, sim)
        
#==============================================================================
# kmeans++ initilization, euclidean distance similarity
#==============================================================================
print '.....do kmeans for kmeans++ initilization with euclidean distance similarity.....'
#resultFile.write('.....do kmeans for kmeans++ initilization with euclidean similarity.....\n')
for i in range(0, len(Clist)):
    for j in range(0, len(cnvgList)):
        filePath = 'docCluster_eucliSim_' + str(Clist[i]) + '_cnvg_' + str(cnvgList[j]) + '.txt'
        runClustering(Clist[i], cnvgList[j], filePath, 
                      devdata, kmeansPlusPlus, euclideanSim)

#==============================================================================
# kmeans++ initilization, cosine distance similarity, use tf * idf as weight
#==============================================================================
print '.....do kmeans for kmeans++ initilization with cosine similarity, tf * idf as weight.....'
#resultFile.write('.....do kmeans for kmeans++ initilization with euclidean similarity.....\n')
for i in range(0, len(Clist)):
    for j in range(0, len(cnvgList)):
        filePath = 'docCluster_ifIdf_' + str(Clist[i]) + '_cnvg_' + str(cnvgList[j]) + '.txt'
        runClustering(Clist[i], cnvgList[j], filePath, 
                      devdata, kmeansPlusPlus, sim, True)


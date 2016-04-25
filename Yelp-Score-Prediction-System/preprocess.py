# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 21:51:33 2016

This script provides basic functions to do data pre-processing for the JSON raw
Yelp dataset. It contains function to transform raw dataset to a temporary data
file which contains the token strings for each sample after the stop words removal
and data clearing. It also contains function to generate dictionary based on 
different feature selection methdos. It also contains function to generate 
sparse matrix based on temporaty data files for use in training.

@author: minxu
"""
import os.path
import json
import re
import string
import heapq
import scipy as sp
from scipy import sparse
import numpy as np

train_d_size = 1255353
f_size = 4000
w_size = f_size + 1

regex0 = re.compile('[^\x00-\x7F]') #regex for non-ascii
regex1 = re.compile('[%s]' % re.escape(string.punctuation)) #regex for punct
regex2 = re.compile('.*[0-9].*') #regex for string with numbers

#==============================================================================
# parse stop list file
#==============================================================================
def processStopList(stoplistfile):
    stoplist = set()
    with open(stoplistfile) as sl:
        for l in sl:
            [stoplist.add(s) for s in l.split()]
    return stoplist

#==============================================================================
# parse training set, clean punctuations and remove stop words
# generate a smaller output file for future processing
#==============================================================================
def processTrain(inputfile, outputfile, stoplist):
    newjson = open(outputfile, 'w')
    with open(inputfile) as j:
        for l in j:
            d = json.loads(l)
            d['text'] = tokenize(d['text'], stoplist)
            newjson.write(d['stars'].__str__() + "\t" + d['text'] + "\n")
    newjson.close()

#==============================================================================
# parse dev or test file and do the same cleaning for text
# generate lists of samples with review_id and cleaned text
#==============================================================================
def processTest(inputfile, stoplist):
    reivewidList = []
    textList = []
    with open(inputfile) as j:
        for l in j:
            d = json.loads(l)
            d['text'] = tokenize(d['text'], stoplist)
            reivewidList.append(d['review_id'])
            textList.append(d['text'])
            
    return reivewidList, textList

#==============================================================================
# tokenize the text by removing punctuations, stop words and numbers
#==============================================================================
def tokenize(textstring, stoplist):
     textstring = regex0.sub(" ", textstring)
     textstring = regex1.sub("", textstring).lower()
     textstring = string.join([s for s in textstring.split() if s not in stoplist and not bool(regex2.search(s))])
     return textstring

#==============================================================================
# do analysis on files after preprocessing, calculate token frequency 
# and distributions of stars, generate dictionary files (token vs index) and
# the training examples file with index based text. Notice that each file will
# have two versions based on ctf-based feature selection and df-based
#==============================================================================
def preAnalysis(inputfile, outputfile_ctf, outputfile_df, dictFile_ctf, dictFile_df):
    distri = [0] * 5 #stars distributions
    tokenCtf = dict() #token ctf
    tokenDf = dict() #token df
    with open(inputfile) as f:
        docid = 0
        for l in f:
            l = l.split('\t')
            distri[int(l[0])-1] = distri[int(l[0])-1] + 1;            
            for s in l[1].split():
                if s in tokenCtf: #ctf based feature counts
                    tokenCtf[s] = tokenCtf[s] + 1
                else:
                    tokenCtf[s] = 1
                if s in tokenDf: #df based feature counts
                    tokenDf[s].add(docid)
                else:
                    newset = set()
                    newset.add(docid)
                    tokenDf[s] = newset
            docid = docid + 1
    
    #convert size of set to document frequency
    tokenDf_cnt = dict()
    for token in tokenDf:
        tokenDf_cnt[token] = len(tokenDf[token])
    
    #select the top 2000 features based on ctf and df
    topCtflist = heapq.nlargest(f_size, tokenCtf, key = tokenCtf.get)
    topDflist = heapq.nlargest(f_size, tokenDf_cnt, key = tokenDf_cnt.get)
    
    for i in range(0, 10):
        print topCtflist[i] +  ': ' + str(tokenCtf[topCtflist[i]])
        
    for i in range(0, 5):
        print 'star ' + str(i+1) + ' : ' + str(distri[i])
    
    #generate dictionary based on ctf feature
    index = 0
    tokenDict_ctf = dict()
    with open(dictFile_ctf, 'w') as f:
        for t in topCtflist:
            tokenDict_ctf[t] = index
            f.write(t + ' ' + str(index) + '\n')
            index = index + 1
    
    #generate dictionary based on df feature
    index = 0
    tokenDict_df = dict()
    with open(dictFile_df, 'w') as f:
        for t in topDflist:
            tokenDict_df[t] = index
            f.write(t + ' ' + str(index) + '\n')
            index = index + 1
    
    #generate the training set in forms of token id, based on both feature
    #selection methods
    ofctf = open(outputfile_ctf, 'w')
    ofdf = open(outputfile_df, 'w')
    with open(inputfile) as f:
        for l in f:
            l = l.split('\t')
            ofctf.write(l[0] + '\t')
            ofdf.write(l[0] + '\t')
            for s in l[1].split():
                if s in tokenDict_ctf:
                    ofctf.write(str(tokenDict_ctf[s]) + ' ')
                if s in tokenDict_df:
                    ofdf.write(str(tokenDict_df[s]) + ' ')
            ofctf.write('\n')
            ofdf.write('\n')
            
    ofctf.close()
    ofdf.close()
    
    return tokenDict_ctf, tokenDict_df
    
#==============================================================================
# parse the dictionary file to a ditionary with token as key and id as value
#==============================================================================
def parseDictionary(dictFile):
    dictionary = dict()
    with open(dictFile) as f:
        for l in f:
            l = l.split()
            dictionary[l[0]] = int(l[1])
    return dictionary
    
#==============================================================================
# extract features and formulate sparse matrices based on preprocessed data
#==============================================================================
def featureExtraction(vectorFile):
    dwMatrix = sp.sparse.lil_matrix((train_d_size, w_size))
    score = np.matrix(np.zeros((train_d_size, 1)))

    with open(vectorFile) as f:
        train_id = 0
        for l in f:
            l = l.split('\t')
            score[train_id, 0] = int(l[0])
            dwMatrix[train_id, 0] = 1
            for w in l[1].split():
                dwMatrix[train_id, int(w) + 1] = dwMatrix[train_id, int(w) + 1] + 1
            train_id = train_id + 1
            
        dwMatrix = dwMatrix.tocsc()
        dfVector = (dwMatrix != 0).sum(0)
        ctfVector = dwMatrix.sum(0)
        
        return (score, dwMatrix, dfVector, ctfVector)

#==============================================================================
# cutomizing features and formulate sparse matrices based on preprocessed data
# use tf-idf as the element values
#==============================================================================
def featureExtraction_custom(dwMatrix, dfVector):
    N, f = dwMatrix.shape
    idfVector = sp.log(N / dfVector + 1)
    #idfVector = np.ones((N, 1)) * idfVector
    #return sp.sparse.csr_matrix(dwMatrix.multiply(idfVector))
    dwMatrix_custom = dwMatrix.tolil()
    for i in range(0, N):
        print i
        dwMatrix_custom[i, :] = dwMatrix_custom[i, :].multiply(idfVector)
    return dwMatrix_custom.tocsr()
    
#==============================================================================
# extract features and formulate sparse matrices for test and dev data 
#==============================================================================
def featureExtraction_test(testdata, dictionary):
    reviewidList = testdata[0]
    textList = testdata[1]
    
    dwMatrix = sp.sparse.lil_matrix((len(reviewidList), w_size))
    for i in range(0, len(textList)):
        tf_vector = [0] * w_size
        tf_vector[0] = 1
        t = textList[i].split()        
        for s in t:
            if s in dictionary:
                tf_vector[dictionary[s] + 1] = tf_vector[dictionary[s] + 1] + 1
        dwMatrix[i, :] = sp.sparse.lil_matrix(tf_vector)
    
    dfVector = (dwMatrix != 0).sum(0)
    ctfVector = dwMatrix.sum(0)
    return dwMatrix, dfVector, ctfVector

#==============================================================================
# main routine, return four variables, two of them are dictionaries for token,
# the other two are devdata and testdata, which are tuples and each of them
# contains two lists: a reviewid list and a text list
#==============================================================================
def main():
    stoplist = processStopList('stopword.list')
    if not os.path.exists(os.getcwd() + '\\newtrain'):
        processTrain('yelp_reviews_train.json', 'newtrain', stoplist)
        
    if not os.path.exists(os.getcwd() + '\\train_preprocessed_ctf'):
        tokenDict_ctf, tokenDict_df = preAnalysis('newtrain', 'train_preprocessed_ctf', 'train_preprocessed_df', 'token_dictionary_ctf', 'token_dictionary_df')
    else:
        tokenDict_ctf = parseDictionary('token_dictionary_ctf')
        tokenDict_df = parseDictionary('token_dictionary_df')
    
    devdata = processTest('yelp_reviews_dev.json', stoplist)
    testdata = processTest('yelp_reviews_test.json', stoplist)
    
    return tokenDict_ctf, tokenDict_df, devdata, testdata
    
if __name__ == "__main__":
    main()
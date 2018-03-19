#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 02:31:09 2018

@author: kaushal
"""

import math
from textblob import TextBlob as tb
import sys

import pandas as pd

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)
    
def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

#df=pd.read_csv("/Users/kaushal/Desktop/Trinity/DCU/QueryResults.csv")
docPath= str(sys.argv[1])
print(docPath)
df=pd.read_csv(docPath)

body= df.iloc[:,8]
Id= df.iloc[:,0]

"""
str= body[1]
for i in range(10):
    str= str.replace(rem[i],'')
"""
rem= ['<code>','<pre>','</code>','</pre>','<strong>','</strong>','<em>','</em>','<p>','/p']
bloblist= []

for i in range(10000):
    str= body[i]
    for i in range(10):
        str= str.replace(rem[i],'')
    strblob= tb(str)
    bloblist.append(strblob)

for j in range(100):
    bloblist= []
    str= body[j]
    for i in range(10):
        str= str.replace(rem[i],'')
    strblob= tb(str)
    bloblist.append(strblob)

    #bloblist = [document1] #, document2, document3]

    for i, blob in enumerate(bloblist):
        print("Top words for user id {}".format(Id[i]))
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        #print(sorted_words)
        for word, score in sorted_words[:10]:
            print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
    
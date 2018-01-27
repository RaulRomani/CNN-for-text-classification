# -*- coding: utf-8 -*-

import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
from embedding_utils import *

def get_word2vec_encode(word):
    if word in m_word2vec:
        return m_word2vec[word]
    return None

def get_glove_encode(word):
    if word in m_glove:
        return m_glove[word]

def get_fasttext_encode(word):
    if word in m_fasttext:
        return m_fasttext[word]

def build_sentiment_dataset_from_xml(xml_path):
    SENTIMENT_POSITIVE = 'P'
    SENTIMENT_NEGATIVE = 'N'
    SENTIMENT_NONE = 'NONE'
    SENTIMENT_NEUTRAL = 'NEU'

    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for tweet in root.findall('tweet'):
        temp = []
        content = tweet.find('content').text
        content = content.encode("utf-8")
        content = clean_str(content)
        temp.append(content)
        polarity = tweet.find('sentiment').find('polarity').find('value').text
        if polarity == SENTIMENT_POSITIVE:
            temp.append(2)
        elif polarity == SENTIMENT_NEGATIVE:
            temp.append(0)
        elif polarity == SENTIMENT_NEUTRAL:
            temp.append(1)
        else :
            temp.append(3)
        data.append(temp)
    return data

def build_sentiment_dataset_test_from_xml(xml_path,qrel_path):
    qrel = file(qrel_path,'r')
    SENTIMENT_POSITIVE = 'P'
    SENTIMENT_NEGATIVE = 'N'
    SENTIMENT_NONE = 'NONE'
    SENTIMENT_NEUTRAL = 'NEU'

    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for tweet in root.findall('tweet'):
        temp = []
        content = tweet.find('content').text
        content = content.encode("utf-8")
        content = clean_str(content)
        temp.append(content)
        #polarity = tweet.find('sentiment').find('polarity').find('value').text
        polarity = qrel.readline().split()[1]
        if polarity == SENTIMENT_POSITIVE:
            temp.append(2)
        elif polarity == SENTIMENT_NEGATIVE:
            temp.append(0)
        elif polarity == SENTIMENT_NEUTRAL:
            temp.append(1)
        else :
            temp.append(3)
        data.append(temp)
    qrel.close()
    return data

#%% Get batch for training
# Call Example 
# batch,sentiment = getBatch(data,1,10)
def get_sentiment_batch(data,start,end):
    #max_s = 0
    max_size = 19
    batch = []
    sentiment = []
    for b in range(start,end):
        dim_1 = []
        dim_2 = []
        dim_3 = []
        content,s = data[b]
        words = content.split(' ')
        sentence = []
        for w in words:
            d_1 = get_word2vec_encode(w)
            d_2 = get_glove_encode(w)
            d_3 = get_fasttext_encode(w)
            embed = []
            if d_1 is not None and d_2 is not None and d_3 is not None:
                embed.append(d_1)
                embed.append(d_2)
                embed.append(d_3)
                sentence.append(np.array(embed).transpose())
        if sentence:
            #if max_s < len(sentence):
            #    max_s = len(sentence)
            #    print 'LS',max_s
            if len(sentence) < max_size:
                deff = max_size - len(sentence)
                embed = []
                embed.append([0.0]*300)
                embed.append([0.0]*300)
                embed.append([0.0]*300)
                for i in range(deff):
                    sentence.append(np.array(embed).transpose())
            batch.append(sentence)
            sentiment.append(s)
    return np.array(batch),np.array(sentiment)

m_fasttext = load_word2vec('embeddings_models/wiki.es_ligth.vec',binary = False)
m_word2vec = load_word2vec('embeddings_models/SBW-vectors-300-min5_ligth',binary = False)
m_glove = load_word2vec('embeddings_models/glove_combine_ligth',binary = False)

#data = build_sentiment_dataset_test_from_xml('datasets/intertass-test.xml','datasets/intertass-sentiment.qrel')
#batch,sentiment = get_sentiment_batch(data,0,1898)
#print sentiment.shape, batch.shape

#data = build_sentiment_dataset_from_xml('datasets/intertass-train-tagged.xml')
#batch,sentiment = get_sentiment_batch(data,0,1008)
#print sentiment.shape, batch.shape

#data = build_sentiment_dataset_from_xml('datasets/intertass-development-tagged.xml')
#batch,sentiment = get_sentiment_batch(data,0,506)
#print sentiment.shape, batch.shape
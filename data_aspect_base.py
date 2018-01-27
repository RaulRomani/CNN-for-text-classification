from gensim.models import Word2Vec
import numpy as np
from embedding_utils import *

def load_word2vec(path,binary = False):
    from gensim.models.keyedvectors import KeyedVectors
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=binary)
    return word_vectors

def get_word2vec_encode(word):
    if word in m_word2vec:
        return m_word2vec[word]
    return None

def get_glove_encode(word):
    if word in m_glove:
        return m_glove[word]
    return None

def get_fasttext_encode(word):
    if word in m_fasttext:
        return m_fasttext[word]
    return None

def remove_tags(xml_str):
    n_str = ''
    ommit = False
    for c in xml_str:
        if c == '<':
            ommit = True
        elif c == '>':
            ommit = False
        else:
            if not ommit:
                n_str = n_str + c
    return n_str

def build_aspect_base_dataset_from_xml(xml_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for tweet in root.findall('tweet'):
        content = clean_str(remove_tags(ET.tostring(tweet,encoding='utf-8')))
        for s in tweet.findall('sentiment'):
            temp = []
            temp.append(content)
            aspect = s.get('aspect') + ' ' + s.get('entity')
            aspect = aspect.lower().replace('_',' ')
            aspect = aspect.replace('|',' ')
            temp.append(aspect)
            temp.append(get_sentiment(s.get('polarity')))
            data.append(temp)
    return data

def get_sentiment(polarity):
    SENTIMENT_POSITIVE = 'P'
    SENTIMENT_NEGATIVE = 'N'
    SENTIMENT_NONE = 'NONE'
    SENTIMENT_NEUTRAL = 'NEU'
    if polarity == SENTIMENT_POSITIVE:
        return 2
    elif polarity == SENTIMENT_NEGATIVE:
        return 0
    elif polarity == SENTIMENT_NEUTRAL:
        return 1
    else :
        return 3

#%% Get batch for training
# Call Example 
# batch,sentiment = getBatch(data,1,10)
def get_aspect_base_batch(data,start,end):
    #max_s = 0
    max_size = 19
    batch = []
    sentiment = []
    for b in range(start,end):
        dim_1 = []
        dim_2 = []
        dim_3 = []
        content,aspects,s = data[b]
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
            #else:
            #   print 'Not found',w
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
            dim_1 = [0.0]*300
            dim_2 = [0.0]*300
            dim_3 = [0.0]*300
            n_aspects = 0
            for a in aspects.split():
                d_1 = get_word2vec_encode(a)
                d_2 = get_glove_encode(a)
                d_3 = get_fasttext_encode(a)
                if d_1 is not None and d_2 is not None and d_3 is not None:
                    dim_1 = d_1 + get_word2vec_encode(a)
                    dim_2 = d_2 + get_glove_encode(a)
                    dim_3 = d_3 + get_fasttext_encode(a)
                    n_aspects = n_aspects + 1
            if n_aspects > 0:
                embed = []
                embed.append(np.array(dim_1)/n_aspects)
                embed.append(np.array(dim_2)/n_aspects)
                embed.append(np.array(dim_3)/n_aspects)
                sentence.append(np.array(embed).transpose())
                batch.append(sentence)
                sentiment.append(s)
    return np.array(batch),np.array(sentiment)

m_fasttext = load_word2vec('embeddings_models/wiki.es_ligth.vec',binary = False)
m_word2vec = load_word2vec('embeddings_models/SBW-vectors-300-min5_ligth',binary = False)
m_glove = load_word2vec('embeddings_models/glove_combine_ligth',binary = False)

#data      = build_aspect_base_dataset_from_xml('datasets/stompol-train-tagged.xml')
#batch,sentiment = get_aspect_base_batch(data,0,784)
#print batch.shape, sentiment.shape
#data_test = build_aspect_base_dataset_from_xml('datasets/stompol-test-tagged.xml')
#batch,sentiment = get_aspect_base_batch(data_test,0,500)
#print batch.shape, sentiment.shape
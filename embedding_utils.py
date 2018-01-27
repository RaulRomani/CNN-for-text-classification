# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
import re

stop = stopwords.words('spanish')

def clean_dataset_for_embedings(path,output):
    i = open(path,'r')
    o = open(output,'w')
    for w in range(TRAINING_SENTENCES):
        if w % 100 == 0:
            print w*1.0/TRAINING_SENTENCES*100,'%'
        s = i.readline()
        if len(s) > 6 :
            s = clean_str(s)
            o.write(s)
    i.close()
    o.close()

def load_word2vec(path,binary = False):
    from gensim.models.keyedvectors import KeyedVectors
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=binary)
    #print word_vectors['nada']
    return word_vectors

def clean_str(str_input):

    str_input = str_input.lower()
    text = [w for w in str_input.split() if w not in stop]
    str_input = ''
    for w in text:
        str_input = str_input + ' ' + w

    str_accent =  ['Ño' ,'ño' ,'á','é','è','í','ó','ú','Á','É','è','Í','Ó','Ú','ñ','Ñ','%','#','@','"',"'",'/','-','°','(',')','[',']','.',',',':',';','ç','Ò','²','«','»','!','¡','¿','?','*','^','=']
    str_replace = ['nio','nio','a','e','e','i','o','u','a','e','e','i','o','u','n','n','' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'c','o','2','' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ,'' ]
    #str_input = str_input.encode("utf-8")
    for s in range(len(str_accent)):
        str_input = str_input.replace(str_accent[s],str_replace[s])
    
    return re.sub(' +',' ',str_input).rstrip()

def train_word2vec(path):
    i = open(path,'r')
    data = []
    for l in range(TRAINING_SENTENCES):
        s = i.readline()
        data.append(s.split(' '))
    i.close()
    model = Word2Vec(data, size=300, window=5, min_count=1, workers=4)
    model.wv.save_word2vec_format('embeddings_models/model_word2vec_'+str(TRAINING_SENTENCES),binary = False)
    return model

def train_fasttext(path):
    # https://pypi.python.org/pypi/fasttext
    import fasttext
    model = fasttext.skipgram(path, 'embeddings_models/model_fasttext_'+str(TRAINING_SENTENCES),dim=300)
    
def train_glove(path):
    import itertools
    from gensim.models.word2vec import Text8Corpus
    from gensim.scripts.glove2word2vec import glove2word2vec
    from glove import Corpus, Glove
    #import os
    #import struct
    sentences = list(itertools.islice(Text8Corpus(path),None))
    corpus = Corpus()
    corpus.fit(sentences, window=10)
    glove = Glove(no_components=300, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    file_name = 'embeddings_models/model_glove_'+str(TRAINING_SENTENCES)
    glove.save(file_name)
    glove2word2vec(file_name, file_name +'_modified')
    """
    command = 'python -m gensim.scripts.glove2word2vec -i ' +file_name+' -o '+file_name+'_modified'
    os.system(command)
    with open(file_name+'_modified', mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        print 'Content',fileContent
    """
    print 'Finished'
    return glove

def load_fast_text(path):
    import fasttext
    model = fasttext.load_model(path, encoding='utf-8')
    return model

def load_glove(path):
    #python -m gensim.scripts.glove2word2vec -i model_glove_1000 -o model_glove_1000_modified
    #return Word2Vec.load(path)
    #return load_word2vec(path,binary = True)
    from gensim.models.keyedvectors import KeyedVectors
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=True)
    return word_vectors

def train_embeddings():
    clean_dataset_for_embedings('/home/alonzo/Documentos/Projects/wikipedia_dataset.txt',TRAINING_FILE)
    print 'Start fasttext'
    train_fasttext(TRAINING_FILE)
    #print 'Start glove'
    #train_glove(TRAINING_FILE)
    print 'Start word2vec'
    train_word2vec(TRAINING_FILE)
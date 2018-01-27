from data_sentiment import *
from data_aspect_base import *

m_fasttext = load_word2vec('embeddings_models/wiki.es_ligth_2.vec',binary = False)
m_word2vec = load_word2vec('embeddings_models/SBW-vectors-300-min5_ligth_2',binary = False)
#m_glove = load_word2vec('embeddings_models/glove_combine_ligth',binary = False)
#model = m_fasttext

vocabulary = {}
def add_to_vocabulary_from_sentiment(vocabulary,data,model):
    for d,s in data:
        words = d.split()
        for w in words:
            if w not in vocabulary:
                if w in model:
                    vocabulary[w] = model[w]
                else:
                    print 'NF',w

def add_to_vocabulary_from_aspect_base(vocabulary,data,model):
    for d,a,s in data:
        words = d.split()
        for w in words:
            if w not in vocabulary:
                if w in model:
                    vocabulary[w] = model[w]
                else:
                    print 'NF',w
        words = a.split()
        for w in words:
            if w not in vocabulary:
                if w in model:
                    vocabulary[w] = model[w]
                else:
                    print 'NF',w

def save_new_model_from_vocabulary(vocabulary,path):
    model = file(path,'w')
    keys = vocabulary.keys()
    model.write(str(len(keys))+' 300\n')
    for w in vocabulary:
        model.write(w)
        w_v = vocabulary[w]
        for e in w_v:
            model.write(' '+str(e))
        model.write('\n')
    model.close()

def combine_models(model_1,model_2):
    new_model = {}
    keys = []
    for w in model_1.vocab:
        keys.append(w)
    for w in keys:
        if w in model_1 and w in model_2:
            new_model[w] = (model_1[w] + model_2[w] )/ 2
    return new_model

model = combine_models(m_word2vec,m_fasttext)

add_to_vocabulary_from_sentiment(vocabulary,build_sentiment_dataset_from_xml('datasets/intertass-train-tagged.xml'),model)
add_to_vocabulary_from_sentiment(vocabulary,build_sentiment_dataset_from_xml('datasets/intertass-development-tagged.xml'),model)
add_to_vocabulary_from_sentiment(vocabulary,build_sentiment_dataset_from_xml('datasets/intertass-test.xml'),model)


add_to_vocabulary_from_aspect_base(vocabulary,build_aspect_base_dataset_from_xml('datasets/stompol-train-tagged.xml'),model)
add_to_vocabulary_from_aspect_base(vocabulary,build_aspect_base_dataset_from_xml('datasets/stompol-test-tagged.xml'),model)

save_new_model_from_vocabulary(vocabulary,'embeddings_models/glove_combine_ligth_2')



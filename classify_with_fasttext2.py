#################################################################################
########### NEEDS TO BE RUN WITH AI FABRIC ENV ###################################
#################################################################################

import numpy as np
import pandas as pd
import fasttext as ft2
import gensim
from sklearn.datasets import fetch_20newsgroups

from sklearn.preprocessing import StandardScaler


from time import time
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from sklearn.svm import  SVC
import nltk


####################   HELPER FUNCTIONS ################################
my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
english_words = set(nltk.corpus.words.words())
english_words.remove('ax')

my_model_path= "model\\myFastTextModel2_skipgram_100_reduced.bin"
RemoveStopWords = True

def simple_preprocess_func (doc, bRemoveStopWords):
    document =doc
    if bRemoveStopWords:
        document = gensim.parsing.remove_stopwords(document)

    return ' '.join([w for w in gensim.utils.simple_preprocess(document, min_len=2, max_len=25) if w in english_words])


def read_corpus(corpus, bRemoveStopWords):
    # with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
    for i, doc in enumerate(corpus):
        ret_doc = simple_preprocess_func(doc, bRemoveStopWords)
        len_document = len (ret_doc.split())
        if len_document >3:
            yield ret_doc



######################## FAST TEXT ############################################
def generate_training_corpus():
    newsgroups_train = fetch_20newsgroups(subset='train',
                                  remove=('headers', 'footers', 'quotes'),
                                  categories = my_cats)
    training_corpus_path='..\\..\\data_set\\20newsgroups\\training_corpus_2.txt'


    with open(training_corpus_path, mode='w') as myfile:
        for doc in newsgroups_train['data']:
            clean_line = simple_preprocess_func(doc=doc, bRemoveStopWords=RemoveStopWords)
            myfile.write(clean_line)
            myfile.write('\n')

def train_fast_text():
    # # # # # # TRAIN THE MODEL # # # # # # 
    train_corpus_path = '..\\..\\data_set\\20newsgroups\\training_corpus_2.txt' 
    t0 = time()
    print ('starting training of fasttext model...')
    model = ft2.train_unsupervised(input=train_corpus_path, model='skipgram', verbose=0, dim=100, epoch=10)
    print ('training of fastText done in {}'.format (time() - t0))


    model.save_model(my_model_path)

def generate_ft_data_set(subset='train', dim=100):
    data_set_list = [list(read_corpus(fetch_20newsgroups(subset=subset,
                                                            remove=('headers', 'footers', 'quotes'),
                                                            categories=[cat])['data'],
                                     bRemoveStopWords=RemoveStopWords))\
                        for cat in my_cats]

    corpus = [doc for categroy_list in data_set_list for doc in categroy_list ]

    # print ('raw corpus size : {}'.format(len(corpus)))
    categories_lengths=[len(cat_liste) for cat_liste in data_set_list]
    categories = [[k for _ in range(0,length)] for k,length in enumerate(categories_lengths)]
    cats = [cat for elem_list in categories for cat in elem_list]  
    y = np.array(cats)


    model = ft2.load_model(my_model_path)
    t0 = time()
    X = np.vstack([model.get_sentence_vector(doc) for doc in corpus])
    print(f'it took {time()- t0} seconds to infer fasttext embeddings')

    print (f'subset {subset} shape {X.shape}:')

    np.savetxt('fastText_data\\X{}_{}.csv'.format(dim, subset), X, delimiter=",")
    np.savetxt('fastText_data\\y{}_{}.csv'.format(dim, subset), y, delimiter=",")


def train_classifier(dim=100):
    # my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
    # X_train, y_train = generate_data_set(my_cats, subset='train')
    # X_test, y_test = generate_data_set(my_cats, subset='test')
    X_train = np.loadtxt(f'fastText_data\\X{dim}_train.csv' , delimiter=',')
    y_train = np.loadtxt(f'fastText_data\\y{dim}_train.csv' , delimiter=',')
    X_test = np.loadtxt(f'fastText_data\\X{dim}_test.csv' , delimiter=',')
    y_test = np.loadtxt(f'fastText_data\\y{dim}_test.csv' , delimiter=',')

    t0 = time()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f'scaling the data it took {time()- t0} seconds')

    clf = SVC(probability=True)
    t0 = time()
    clf.fit(X_train, y_train)
    print(f'training the model {clf.__class__.__name__} took {time()- t0} seconds')
    print (f'{clf.__class__.__name__} with FastText score {clf.score(X_test, y_test)}')

    classifier_filename = './model/fasttext_svm_classifier.sav'
    joblib.dump(clf, classifier_filename)

    scaler_filename = "./model/std_scaler.sav"
    joblib.dump(scaler, scaler_filename) 

 


def evaluate_classifier(dim=100):
    # X_train = np.loadtxt(f'fastText_data\\X{dim}_train_reduced.csv' , delimiter=',')
    # y_train = np.loadtxt(f'fastText_data\\y{dim}_train_reduced.csv' , delimiter=',')
    X_test = np.loadtxt(f'fastText_data\\X{dim}_test.csv', delimiter=',')
    y_test = np.loadtxt(f'fastText_data\\y{dim}_test.csv', delimiter=',')

    # print (f'train corpus shape {X_train.shape}')
    print (f'test corpus shape {X_test.shape}')
    
    # scaler = StandardScaler()
    scaler_filename = "./model/std_scaler.sav"
    scaler = joblib.load(scaler_filename)
    
    t0 = time()
    # X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f'scaling the data it took {time()- t0} seconds')
    
    # index = np.arange(0,X_train.shape[0])
    # np.random.shuffle(index)
    # X_train = X_train[index]
    # y_train = y_train[index]

        # some time later...
    
    # load the model from disk
    classifier_filename = './model/fasttext_svm_classifier.sav'
    clf = joblib.load(classifier_filename)

    # clf = SVC(probability=True)
    # t0 = time()
    # clf.fit(X_train, y_train)
    # print(f'training the model {clf.__class__.__name__} took {time()- t0} seconds')
    print (f'{clf.__class__.__name__} with FastText accuracy score {clf.score(X_test, y_test)}')
    
    y_pred = clf.predict(X_test)
    print (f'f1 score of {clf.__class__.__name__} is {f1_score(y_test, y_pred, average=None)}')
    print (classification_report(y_test, y_pred))

def plotFasttextSimilarity ():
    X_train = np.loadtxt('fastText_data\\X100_train.csv', delimiter=',')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    print (X_train.shape)

    similarity_matrix = cosine_similarity(X_train)
  
    plt.figure()
    # plt.plot(y_train)

    plt.title('Cosine similarity of fasttext embeddings')
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.show()


if __name__ == "__main__":  
    # generate_training_corpus()
    # train_fast_text()
    # generate_ft_data_set(subset='train', dim=100)
    # generate_ft_data_set(subset='test', dim=100)
    # train_classifier()
    # evaluate_classifier()
    plotFasttextSimilarity()
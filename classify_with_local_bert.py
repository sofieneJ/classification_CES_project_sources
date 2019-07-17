from tools import *

from bert_serving.client import  BertClient

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns

from sklearn.svm import LinearSVC

from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression



cats = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']



def generate_data_set(test_cats, subset='train'):

    data_set_list = [list(read_corpus(fetch_20newsgroups(subset=subset,
                                                            remove=('headers', 'footers', 'quotes'),
                                                            categories=[cat])['data'],
                                           tokens_only=True, bRemoveStopWords=True, bFastText=True))\
                        for cat in test_cats]

    corpus = [doc for categroy_list in data_set_list for doc in categroy_list ]

    print ('{} corpus size : {}'.format(subset, len(corpus)))
    categories_lengths=[len(cat_liste) for cat_liste in data_set_list]
    categories = [[k for _ in range(0,length)] for k,length in enumerate(categories_lengths)]
    cats = [cat for elem_list in categories for cat in elem_list]  
    y = np.array(cats)
    np.savetxt(fname='local_bert_embeddings\\y_{}.csv'.format(subset), X=y, delimiter=',')

    # azure_vm_ip='13.81.122.196'
    local_gpu_machine = '192.168.1.29'
    local_host='localhost'
    with BertClient(ip=local_host, port=5555, port_out=5556, show_server_config=True, timeout=-1) as bc:
        # print (bc.pending_request)
        # print (bc.status)
        # embeddings = bc.encode(['I like paragliding', 'I love kitesurf'],blocking=True, is_tokenized=False)
        # print (embeddings)
        t0 = time()
        X = np.vstack([bc.encode([doc.split() for doc in corpus[256*i:min((i+1)*256,len(corpus))]], is_tokenized=True) for i in range(0,int(len(corpus)/256)+1)])
    
        print (f'embedding generated in {time() - t0} seconds')
        print ('subset {} shape {}:'.format(subset,X.shape))
        np.savetxt(fname='local_bert_embeddings\\X_{}.csv'.format(subset), X=X, delimiter=',')

    return X, y

def plotBERTTextSimilarity ():
    X_train = np.loadtxt('local_bert_embeddings\\X_train.csv', delimiter=',')
    # y_train = np.loadtxt('local_bert_embeddings\\y_train.csv', delimiter=',')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)

    similarity_matrix = np.inner(X_train, X_train)
    # nb_docs = X_train.shape[0]
    # similarity_matrix = np.zeros(shape=(nb_docs,nb_docs))
    # for i in range (0, nb_docs):
    #     for j in range (0, nb_docs):
    #     similarity_matrix[i,j]=cosine_similarity(X_train[i,:],X_train[j,:])
    
    # plt.figure()
    # plt.plot(y_train)
    plt.figure()
    plt.title('Cosine similarity of BERT Base representation')
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.show()


def classify():
    X_train = np.loadtxt('local_bert_embeddings\\X_train.csv', delimiter=',')
    y_train = np.loadtxt('local_bert_embeddings\\y_train.csv', delimiter=',')
    X_test = np.loadtxt('local_bert_embeddings\\X_test.csv', delimiter=',')
    y_test = np.loadtxt('local_bert_embeddings\\y_test.csv', delimiter=',')

    print (X_train.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    index = np.arange(0,X_train.shape[0])
    np.random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]

    clf = knn()
    clf.fit(X_train, y_train)
    print (f'{clf.__class__.__name__} with ELMo reps score {clf.score(X_test, y_test)}' )
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print ('LogisticRegression with BERT Base score %f' % clf.score(X_test, y_test))
    clf = MLPClassifier(hidden_layer_sizes=(1000,1000))
    clf.fit(X_train, y_train)
    print ('MLPClassifier with BERT Base score %f' % clf.score(X_test,y_test))
    clf = SVC()
    clf.fit(X_train, y_train)
    print ('SVC with BERT Base score %f' % clf.score(X_test,y_test))


def compute_f1_score():
    X_train = np.loadtxt('local_bert_embeddings\\X_train.csv', delimiter=',')
    y_train = np.loadtxt('local_bert_embeddings\\y_train.csv', delimiter=',')
    X_test = np.loadtxt('local_bert_embeddings\\X_test.csv', delimiter=',')
    y_test = np.loadtxt('local_bert_embeddings\\y_test.csv', delimiter=',')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    index = np.arange(0,X_train.shape[0])
    np.random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]

    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print (f'{clf.__class__.__name__} with local Bert score {clf.score(X_test, y_test)}')

    print (f'f1 score of {clf.__class__.__name__} is {f1_score(y_test, y_pred, average=None)}')


if __name__ == "__main__":  
    my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
    generate_data_set(my_cats, subset='train')
    generate_data_set(my_cats, subset='test')
    # classify()
    # plotBERTTextSimilarity()
    compute_f1_score()
    # draw_tsne('local_bert_embeddings\\X_train.csv', 'local_bert_embeddings\\y_train.csv', method='BERT')
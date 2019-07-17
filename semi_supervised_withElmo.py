from tools import *
from elmoformanylangs import Embedder
import random 
import pickle

import sklearn.metrics as met
import os

my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']

def generate_data_set(test_cats=my_cats, subset='train'):
    dataset_list = [list(read_corpus(fetch_20newsgroups(subset=subset,
                                            remove=('headers', 'footers', 'quotes'),
                                        categories=[cat])['data'],
                                        tokens_only=True,  bFastText=False, bRemoveStopWords=True))\
                            for cat in test_cats]
    corpus = [doc for categroy_list in dataset_list for doc in categroy_list ]

    categories_lengths=[len(cat_liste) for cat_liste in dataset_list]
    categories = [[k for _ in range(0,length)] for k,length in enumerate(categories_lengths)]

    cats = [cat for elem_list in categories for cat in elem_list]  
    y = np.array(cats)
    np.savetxt('ELmo_20news_group_rep\\y_train_reduced.csv', y[range(0,len(cats),50)], delimiter=",")
    
    print ('raw corpus ELMO Rep {} size {}'.format(subset, len(corpus)))
    e = Embedder('..\\PreTrainedElmo_EN', batch_size=64)

    try:
        os.remove('ELmo_20news_group_rep\\X_train_reduced.csv')
    except:
        pass

    with open('ELmo_20news_group_rep\\X_train_reduced.csv', mode='a') as myFile:
        # for i in range(0,2):
        for i in range(0,len(corpus),50):
            em = np.mean(e.sents2elmo([corpus[i]])[0], axis=0)
            # print (em.shape)
            myFile.write('{}'.format(em.tolist()).strip('[').strip(']').replace(' ',''))
            myFile.write('\n')

def classify():

    y = np.loadtxt('ELmo_20news_group_rep\\y_train_reduced.csv', delimiter=',')
    X = np.loadtxt('ELmo_20news_group_rep\\X_train_reduced.csv', delimiter=',')


    centroids = [['car', 'engine', 'drive', 'speed'],
    ['religion', 'jesus', 'god', 'believe', 'heaven', 'sin'],
    ['baseball', 'player', 'run', 'sport', 'hit', 'bat', 'rotation'],
    ['electronics', 'conductive', 'power', 'resistor', 'circuit'],
    ['medical', 'methodology', 'science', 'molecule', 'virus']]


    e = Embedder('..\\PreTrainedElmo_EN', batch_size=64)
    em_vecs = [np.mean(e.sents2elmo(cat_taxo)[0], axis=0) for cat_taxo in centroids]
    

    # X_train = np.loadtxt('custom_doc2vec_data\\X_train.csv', delimiter=',')
    # y_train = np.loadtxt('custom_doc2vec_data\\y_train.csv', delimiter=',')
    # dist = met.pairwise_distances(X= X_train,Y=list_centroids_vectors[0].reshape(1, -1),metric='cosine')
    dist = met.pairwise_distances(X= X,Y=np.vstack(em_vecs), metric='cosine')
    print (dist.shape)
    indexes =  np.argmin(dist, axis=1)

    diff_list = (indexes - y).tolist()
    diff = [1 if d==0 else 0 for d in diff_list]
    print ('taxonomy-based semi supervised classification accuracy : {}'.format(sum(diff)/len(diff))) 


    plt.plot(indexes)

    plt.figure()
    plt.plot(y)

    plt.show()




def plotCustomRepSimilarity ():
    X_train = np.loadtxt('custom_doc2vec_data\\X_train.csv', delimiter=',')
    y_train = np.loadtxt('custom_doc2vec_data\\y_train.csv', delimiter=',')
    train_index = np.where(X_train.any(axis=1))[0]

    X_train = X_train[train_index]
    y_train = y_train[train_index]

    nb_docs = X_train.shape[0]
    similarity_matrix = np.zeros(shape=(nb_docs,nb_docs))
    for i in range (0, nb_docs):
        for j in range (0, nb_docs):
            similarity_matrix[i,j]=cosine_similarity(X_train[i,:],X_train[j,:])
  
    plt.figure()
    plt.plot(y_train)

    plt.title('Cosine similarity of custom document representation')
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.show()

if __name__ == "__main__":  
    # generate_data_set()
    classify()
    # plotCustomRepSimilarity()




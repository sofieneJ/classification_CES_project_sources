from tools import *
from sklearn.feature_extraction.text import TfidfVectorizer
import random 
import pickle

import sklearn.metrics as met



def generate_data_set(test_cats, subset='train'):
    test_dataset_list = [list(read_corpus(fetch_20newsgroups(subset=subset,
                                                            remove=('headers', 'footers', 'quotes'),
                                                            categories=[cat])['data'], 
                                        tokens_only=True, bRemoveStopWords = True, bFastText=False))\
                        for cat in test_cats]

    model_path = "model\\my_doc2vec_20news_model"
    model = gensim.models.doc2vec.Doc2Vec.load(model_path)
    category_size = min ([len(test_dataset_list[k]) for k, _ in enumerate(test_dataset_list)])

    X = np.vstack(np.array(model.infer_vector(test_dataset_list[i][k])) \
        for i, _ in enumerate(test_cats) for k in range(0, category_size) )
    tags = [i for i, _ in enumerate(test_cats) for k in range(0, category_size) ]

    y = np.array(tags)

    print (X.shape)
    print (y.shape)

    return X, y

def classify():

    my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
    my_cats = ['rec.autos', 'soc.religion.christian']
    X_train, y_train = generate_data_set(my_cats, subset='train')

    centroids = [['car', 'engine', 'drive', 'speed'],
    ['religion', 'jesus', 'god', 'believe', 'heaven', 'sin'],
    ['baseball', 'player', 'run', 'sport', 'hit', 'bat', 'rotation'],
    ['electronics', 'conductive', 'power', 'resistor', 'circuit'],
    ['medical', 'methodology', 'science', 'molecule', 'virus']]

    centroids = [['car', 'engine', 'drive', 'speed'],
    ['religion', 'jesus', 'god', 'believe', 'heaven', 'sin']]

    model_path = "model\\my_doc2vec_20news_model"
    model = gensim.models.doc2vec.Doc2Vec.load(model_path)

    centroids_doc_vec_list = [model.infer_vector(doc) for doc in centroids]

    # X_train = np.loadtxt('custom_doc2vec_data\\X_train.csv', delimiter=',')
    # y_train = np.loadtxt('custom_doc2vec_data\\y_train.csv', delimiter=',')
    # dist = met.pairwise_distances(X= X_train,Y=list_centroids_vectors[0].reshape(1, -1),metric='cosine')
    dist = met.pairwise_distances(X= X_train,Y=np.vstack(centroids_doc_vec_list), metric='cosine')
    print (dist.shape)
    indexes =  np.argmin(dist, axis=1)

    diff_list = (indexes - y_train).tolist()
    diff = [1 if d==0 else 0 for d in diff_list]
    print ('taxonomy-based semi supervised classification accuracy : {}'.format(sum(diff)/len(diff))) 


    plt.plot(indexes)

    plt.figure()
    plt.plot(y_train)

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
    # train_word2vec()
    # train_TFIDF()
    # BuildDataSet(subset='train')
    # BuildDataSet(subset='test')
    classify()
    # plotCustomRepSimilarity()




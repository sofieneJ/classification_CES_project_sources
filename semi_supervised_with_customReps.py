from tools import *
from sklearn.feature_extraction.text import TfidfVectorizer
import random 
import pickle

import sklearn.metrics as met



my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']


def train_word2vec ():
    newsgroups_train = fetch_20newsgroups(subset='train',
                                  remove=('headers', 'footers', 'quotes')) 

    train_corpus = list(read_corpus(newsgroups_train['data'], tokens_only=True,
                     bFastText=False, bRemoveStopWords=True))
    
    # Word2Vec model
    # print (train_corpus[0])
    # print (train_corpus[1])
    model = gensim.models.word2vec.Word2Vec(sentences=train_corpus,size=100, compute_loss=True,
                                            min_count=1, max_vocab_size=None, workers=4, sg=1)

    # Build Vocab
    # model.build_vocab(train_corpus)
    print (model.corpus_count)

    # Train the model
    model.train(train_corpus, total_examples=len (train_corpus), epochs=model.epochs)

    #save model
    model_path = 'model\\my_word2vec_20news_model'
    model.save(model_path)
    print (model['funky'])


def train_TFIDF ():
    dataset_list = [list(read_corpus(fetch_20newsgroups(subset='train',
                                          remove=('headers', 'footers', 'quotes'),
                                    categories=[cat])['data'],
                                    bFastText=True, tokens_only=True, bRemoveStopWords=True))\
                        for cat in my_cats]
    train_corpus = [doc for categroy_list in dataset_list for doc in categroy_list ]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_corpus)
    pickle.dump(vectorizer, open('model\\semi_supervised_model\\myTFIDFVectorizer.pkl','wb'))
    
    
    # print (vectorizer.vocabulary_)
    # inv_dic = { indice:word for word, indice in vectorizer.vocabulary_.items()}
    # print (inv_dic[0])
    # print (inv_dic[1])
    # print (train_corpus[0])
    # doc_vec = vectorizer.transform([train_corpus[0]]).todense()
    # liste_index = [i for vv in doc_vec[0]:

    # it = np.nditer(doc_vec, flags=['c_index'])
    # while not it.finished:
    #     if it.value >0.0:
    #         print('{} {} is  {}'.format(it.index, it.value,  inv_dic[it.index] ))
    #     it.iternext()


def aggregate_TFIDF_with_word2ec(vec, vectorizer_dic, word2vec_model):
    ret_vec = np.zeros(word2vec_model.vector_size)
    it = np.nditer(vec, flags=['c_index'])
    while not it.finished:
        word = vectorizer_dic[it.index]
        try:
            word_vec = word2vec_model.wv[word]
            ret_vec = ret_vec+ it.value*word_vec
        except:
            # print ('cannot infer embedding for {}'.format(word))
            # print('{} {} is  {}'.format(it.index, it.value,  vectorizer_dic[it.index] ))
            pass
        it.iternext()

    return ret_vec

def BuildDataSet(subset='train'):
    dataset_list = [list(read_corpus(fetch_20newsgroups(subset=subset,
                                          remove=('headers', 'footers', 'quotes'),
                                    categories=[cat])['data'],
                                    tokens_only=True,  bFastText=True, bRemoveStopWords=True))\
                        for cat in my_cats]
    corpus = [doc for categroy_list in dataset_list for doc in categroy_list ]

    print ('raw corpus size : {}'.format(len(corpus)))
    categories_lengths=[len(cat_liste) for cat_liste in dataset_list]
    categories = [[k for _ in range(0,length)] for k,length in enumerate(categories_lengths)]
    cats = [cat for elem_list in categories for cat in elem_list]  
    y = np.array(cats)

    model_path = "model\\my_word2vec_20news_model"
    w2v_model = gensim.models.word2vec.Word2Vec.load(model_path)

    vectorizer = TfidfVectorizer()
    vectorizer = pickle.load(open('model\\myTFIDFVectorizerTrainSubset.pkl','rb'))
    X = vectorizer.transform(corpus)
    TFIDF_dic = { indice:word for word, indice in vectorizer.vocabulary_.items()}

    res = np.apply_along_axis(func1d=aggregate_TFIDF_with_word2ec, axis=1, arr = X.todense(),
                        vectorizer_dic = TFIDF_dic, word2vec_model=w2v_model)
    
    print ("X_{} shape: {}".format(subset, res.shape))

    np.savetxt('custom_doc2vec_data\\X_{}.csv'.format(subset), res, delimiter=",")
    np.savetxt('custom_doc2vec_data\\y_{}.csv'.format(subset), y, delimiter=",")

def classify():
    centroids = [['car', 'engine', 'drive', 'speed'],
    ['religion', 'jesus', 'god', 'believe', 'heaven', 'sin'],
    ['baseball', 'player', 'run', 'sport', 'hit', 'bat', 'rotation'],
    ['electronics', 'conductive', 'power', 'resistor', 'circuit'],
    ['medical', 'methodology', 'science', 'molecule', 'virus']]

    model_path = "model\\my_word2vec_20news_model"
    w2v_model = gensim.models.word2vec.Word2Vec.load(model_path)

    list_centroids_vectors = []
    for category in centroids:
        cat_vec = np.mean(np.vstack(w2v_model.wv[word] for word in category), axis=0)
        list_centroids_vectors.append(cat_vec)

    X_train = np.loadtxt('custom_doc2vec_data\\X_train.csv', delimiter=',')
    y_train = np.loadtxt('custom_doc2vec_data\\y_train.csv', delimiter=',')
    # dist = met.pairwise_distances(X= X_train,Y=list_centroids_vectors[0].reshape(1, -1),metric='cosine')
    dist = met.pairwise_distances(X= X_train,Y=np.vstack(list_centroids_vectors), metric='cosine')
    # print (dist.shape)
    indexes =  np.argmin(dist, axis=1)

    diff_list = (indexes - y_train).tolist()
    diff = [1 if d==0 else 0 for d in diff_list]
    print ('taxonomy-based semi supervised classification accuracy using custom Reps: {}'.format(sum(diff)/len(diff))) 


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
    # classify()
    # plotCustomRepSimilarity()
    draw_tsne('custom_doc2vec_data\\X_train.csv', 'custom_doc2vec_data\\y_train.csv',
         method='custom rep')




from tools import *
from sklearn.feature_extraction.text import TfidfVectorizer
import random 
import pickle


from sklearn.svm import  SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



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
    t0 = time()
    model.train(train_corpus, total_examples=len (train_corpus), epochs=model.epochs)
    print ('training gensims word2vec in {}'.format (time() - t0))

    #save model
    model_path = 'model\\my_word2vec_20news_model'
    model.save(model_path)
    print ('funky embedding', model['funky'])


def train_TFIDF ():
    dataset_list = [list(read_corpus(fetch_20newsgroups(subset='train',
                                          remove=('headers', 'footers', 'quotes'),
                                    categories=[cat])['data'],
                                    bFastText=True, tokens_only=True, bRemoveStopWords=True))\
                        for cat in my_cats]
    train_corpus = [doc for categroy_list in dataset_list for doc in categroy_list ]

    vectorizer = TfidfVectorizer()
    t0 = time()
    vectorizer.fit(train_corpus)
    print(f'it took {time()- t0} seconds to train TF-IDF')
    pickle.dump(vectorizer, open('model\\myTFIDFVectorizerTrainSubset.pkl','wb'))
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

    t0=time()
    res = np.apply_along_axis(func1d=aggregate_TFIDF_with_word2ec, axis=1, arr = X.todense(),
                        vectorizer_dic = TFIDF_dic, word2vec_model=w2v_model)
    print(f'it took {time()- t0} seconds to build dataset')

    print ("X_{} shape: {}".format(subset, res.shape))

    np.savetxt('custom_doc2vec_data\\X_{}.csv'.format(subset), res, delimiter=",")
    np.savetxt('custom_doc2vec_data\\y_{}.csv'.format(subset), y, delimiter=",")

def classify():
    X_train = np.loadtxt('custom_doc2vec_data\\X_train.csv', delimiter=',')
    y_train = np.loadtxt('custom_doc2vec_data\\y_train.csv', delimiter=',')
    X_test = np.loadtxt('custom_doc2vec_data\\X_test.csv', delimiter=',')
    y_test = np.loadtxt('custom_doc2vec_data\\y_test.csv', delimiter=',')
    train_index = np.where(X_train.any(axis=1))[0]
    test_index = np.where(X_test.any(axis=1))[0]

    # print (y.shape)
    X_train = X_train[train_index]
    y_train = y_train[train_index]
    X_test = X_test[test_index]
    y_test = y_test[test_index]

    print ('X_train shape : {}'.format(X_train.shape))
    print ('X_test shape : {}'.format(X_test.shape))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = knn()
    clf.fit(X_train, y_train)
    print ('knn with custom reps score %f' % clf.score(X_test, y_test))

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print ('LogisticRegression with custom reps score %f' % clf.score(X_test, y_test))
    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    print ('MLPClassifier with custom reps score %f' % clf.score(X_test,y_test))
    clf = SVC()
    clf.fit(X_train, y_train)
    print ('SVC with custom reps score %f' % clf.score(X_test,y_test))
    # plotDoc2VecSimilarity()


def plotCustomRepSimilarity ():
    X_train = np.loadtxt('custom_doc2vec_data\\X_train.csv', delimiter=',')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    print (X_train.shape)

    similarity_matrix = np.inner(X_train, X_train)
  
    plt.figure()
    # plt.plot(y_train)

    plt.title('Cosine similarity of custom document representation')
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.show()

def analyse_accuracy():
    X_train = np.loadtxt('custom_doc2vec_data\\X_train.csv', delimiter=',')
    y_train = np.loadtxt('custom_doc2vec_data\\y_train.csv', delimiter=',')
    X_test = np.loadtxt('custom_doc2vec_data\\X_test.csv', delimiter=',')
    y_test = np.loadtxt('custom_doc2vec_data\\y_test.csv', delimiter=',')
    train_index = np.where(X_train.any(axis=1))[0]
    test_index = np.where(X_test.any(axis=1))[0]

    # print (y.shape)
    X_train = X_train[train_index]
    y_train = y_train[train_index]
    X_test = X_test[test_index]
    y_test = y_test[test_index]

    print ('X_train shape : {}'.format(X_train.shape))
    print ('X_test shape : {}'.format(X_test.shape))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    index = np.arange(0,X_train.shape[0])
    np.random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print (f'{clf.__class__.__name__} with custom doc2vec score {clf.score(X_test, y_test)}')

    my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
    print (classification_report(y_test,y_pred,target_names = my_cats))



if __name__ == "__main__":  
    train_word2vec()
    train_TFIDF()
    BuildDataSet(subset='train')
    BuildDataSet(subset='test')
    # classify()
    # plotCustomRepSimilarity()
    # draw_tsne('custom_doc2vec_data\\X_train.csv', 'custom_doc2vec_data\\y_train.csv', method='custom rep')
    analyse_accuracy()
    # model_path = "model\\my_word2vec_20news_model"
    # w2v_model = gensim.models.word2vec.Word2Vec.load(model_path)
    # print ('cryptozoology embedding ', w2v_model.wv['cryptozoology'])




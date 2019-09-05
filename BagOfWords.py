from tools import *
from sklearn.feature_extraction.text import TfidfVectorizer
import random 
import pickle
from sklearn.model_selection import train_test_split

from sklearn.model_selection import ShuffleSplit

from sklearn.decomposition import NMF

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


my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']

dataset_list = [list(read_corpus(fetch_20newsgroups(remove=('headers', 'footers', 'quotes'),
                                            categories=[cat])['data'],
                                  tokens_only=True, bFastText=True, bRemoveStopWords=True))\
                        for cat in my_cats]
corpus = [doc for categroy_list in dataset_list for doc in categroy_list ]

categories_lengths=[len(cat_liste) for cat_liste in dataset_list]
categories = [[k for _ in range(0,length)] for k,length in enumerate(categories_lengths)]
y = [cat for elem_list in categories for cat in elem_list]


def generate_data_set(test_cats, subset='train'): 
    data_set_list = [list(read_corpus(fetch_20newsgroups(subset=subset,
                                                            remove=('headers', 'footers', 'quotes'),
                                                            categories=[cat])['data'], 
                                        tokens_only=True, bRemoveStopWords = True, bFastText=True))\
                        for cat in test_cats]

    corpus = [doc for categroy_list in data_set_list for doc in categroy_list ]

    # print ('raw corpus size : {}'.format(len(corpus)))
    categories_lengths=[len(cat_liste) for cat_liste in data_set_list]
    categories = [[k for _ in range(0,length)] for k,length in enumerate(categories_lengths)]
    cats = [cat for elem_list in categories for cat in elem_list]  
    y = np.array(cats)

    return corpus, y  

###TEST dataset
# test_dataset_list = [list(read_corpus(fetch_20newsgroups(subset='test',
#                                           remove=('headers', 'footers', 'quotes'),
#                                             categories=[cat])['data'],True))\
#                         for cat in my_cats]
# corpus_test = [doc for categroy_list in test_dataset_list for doc in categroy_list ]

# categories_lengths=[len(cat_liste) for cat_liste in test_dataset_list]
# categories = [[k for _ in range(0,length)] for k,length in enumerate(categories_lengths)]
# y_test = [cat for elem_list in categories for cat in elem_list]


def classify_corpus():
  ###########Training corpus sub-sampling ########################
  # corpus_size = len(y)
  # index_temp = [v for _,v in enumerate(range(0,corpus_size))]
  # random.shuffle(index_temp)
  # train_index = index_temp[0:int (0.9*corpus_size)]
  # test_index = index_temp[int(0.9*corpus_size):corpus_size]
  # train_corpus = [corpus[i] for i in train_index]
  # y_train = [y[i] for i in train_index]
  # test_corpus = [corpus[i] for i in test_index]
  # y_test = [y[i] for i in test_index]

  train_corpus, y_train = generate_data_set(test_cats= my_cats, subset='train')
  test_corpus , y_test = generate_data_set(test_cats=my_cats, subset='test')

  t0 = time()
  ###########TF-IDF on training corpus ########################
  vectorizer = TfidfVectorizer()
  X_train = vectorizer.fit_transform(train_corpus)
  pickle.dump(vectorizer, open('model\\myTFIDFVectorizer.pkl','wb'))
  print (X_train.shape)

  # pickle.load(open('model\\myTFIDFVectorizer.pkl','rb'))

  ###########NMF dimension reduction ########################
  nmf_model = NMF(n_components=20, init='nndsvd', random_state=0)
  np_X_train = np.array(X_train.todense())
  X_train_reduced_dim = nmf_model.fit_transform(np_X_train)
  # np.savetxt('''data\\NMF_tfidf_X_train.csv''', X_train_reduced_dim, delimiter=',')
  print (f'TF-IDF and NMF took {time()-t0}')

  ###########Transforming and predicting Test corpus ########################
  X_test = vectorizer.transform(test_corpus)
  np_X_test = np.array(X_test.todense())
  X_test_reduced_dim = nmf_model.transform(np_X_test)
  print ('test matrice shape : {}'.format(X_test_reduced_dim.shape))

  ###########scaling the data############################
  scaler = StandardScaler()
  X_train_reduced_dim = scaler.fit_transform(X_train_reduced_dim)
  X_test_reduced_dim = scaler.transform(X_test_reduced_dim)

  ###########Train classifier ########################
  print ('training model matrice shape : {}'.format(X_train_reduced_dim.shape))
  
  clf = knn(n_neighbors=5)
  clf.fit(X_train_reduced_dim, y_train)
  print ('predicting with {} on the training set score : {}'.format(clf.__class__.__name__,
                                                      clf.score(X_train_reduced_dim, y_train)))
  
  print ('classification score with classifier {}: {}'.format(clf.__class__.__name__,
                        clf.score(X_test_reduced_dim, y_test)))

  clf = LogisticRegression()
  clf.fit(X_train_reduced_dim, y_train)
  print ('predicting with {} on the training set score : {}'.format(clf.__class__.__name__,
                                                      clf.score(X_train_reduced_dim, y_train)))
  print ('classification score with classifier {}: {}'.format(clf.__class__.__name__,
                        clf.score(X_test_reduced_dim, y_test)))
  
  clf = SVC()
  clf.fit(X_train_reduced_dim, y_train)
  print ('predicting with {} on the training set score : {}'.format(clf.__class__.__name__,
                                                      clf.score(X_train_reduced_dim, y_train)))
  print ('classification score with classifier {}: {}'.format(clf.__class__.__name__,
                        clf.score(X_test_reduced_dim, y_test)))
  
  clf = MLPClassifier()
  clf.fit(X_train_reduced_dim, y_train)
  print ('predicting with {} on the training set score : {}'.format(clf.__class__.__name__,
                                                      clf.score(X_train_reduced_dim, y_train)))
  print ('classification score with classifier {}: {}'.format(clf.__class__.__name__,
                        clf.score(X_test_reduced_dim, y_test)))

  ###########################classification report####################
  y_pred = clf.predict(X_test_reduced_dim)

  print (classification_report(y_test,y_pred,target_names = my_cats))

  # plot_learning_curves(X=X_train_reduced_dim, y=y_train)

###########Plotting LEARNING CURVES ########################
def plot_learning_curves(X,y):
  cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
  
  clf = knn(n_neighbors=5)
  title = "Learning Curves for classifier {}".format(clf.__class__.__name__)
  plot_learning_curve(clf, title, X,y, ylim=(0.1, 1.01), cv=cv, n_jobs=None)

  clf = LogisticRegression()
  title = "Learning Curves for classifier {}".format(clf.__class__.__name__)
  plot_learning_curve(clf, title, X,y, ylim=(0.1, 1.01), cv=cv, n_jobs=None)

  clf = SVC()
  title = "Learning Curves for classifier {}".format(clf.__class__.__name__)
  plot_learning_curve(clf, title, X, y, ylim=(0.1, 1.01), cv=cv, n_jobs=None)

  clf = MLPClassifier()
  title = "Learning Curves for classifier {}".format(clf.__class__.__name__)
  plot_learning_curve(clf, title, X, y, ylim=(0.1, 1.01), cv=cv, n_jobs=None)





  plt.show()


def plotTFIDFSimilarity ():
  vectorizer = TfidfVectorizer()
  X_TFIDF = vectorizer.fit_transform(corpus)
  np_TFIDF = np.array(X_TFIDF.todense())
  
  scaler = StandardScaler()
  np_TFIDF = scaler.fit_transform(np_TFIDF)
  similarity_matrix = np.inner(np_TFIDF, np_TFIDF)

  # nb_docs = np_TFIDF.shape[0]
  # similarity_matrix = np.zeros(shape=(nb_docs,nb_docs))
  # for i in range (0, nb_docs):
  #   for j in range (0, nb_docs):
  #      similarity_matrix[i,j]=cosine_similarity(np_TFIDF[i,:],np_TFIDF[j,:])

  plt.figure()
  plt.title('Cosine similarity of TF-IDF representation')
  plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')


  plt.show()

if __name__ == "__main__":  
  classify_corpus()
  # plotTFIDFSimilarity()

# print (corpus_test[0])
# print(vectorizer.get_feature_names())
# print(X_train.shape)
# print(X_test.shape)
# print(vectorizer.analyzer)
# print (type(X_train[:,vectorizer.vocabulary_['god']]))
# print (X_train[:,vectorizer.vocabulary_['god']].todense())
# print (type(X_train)) // csr_matrix
# print (np_X_train.shape)

# print (newsgroups_train['data'][2])
# print (gensim.utils.simple_preprocess(newsgroups_train['data'][2]))


# english_words = set(nltk.corpus.words.words())
# english_words.remove('ax')
# print (type(english_words))

# myVec = np.vstack(np.arange(k*i,k*i+3) \
#   for i in range(1,3) for k in range(0,7) )
# print (myVec)

# sentence = 'I believe in the existence of paradise'

# my_model_name= "model\\myFT20newsGroupModel.bin"
# model.save_model(my_model_name)

# model = ft.load_model(my_model_name)
# print (sum(model.get_word_vector(w) for w in sentence))
# print (model.get_sentence_vector(sentence))
# print (model.get_word_vector('I'))

# print (model.get_word_vector('I'))
# print (model.get_word_vector('I'))
# print ( len (newsgroups_train['data']))



# cleaned_train_corpus = [clean_document(doc) for doc in newsgroups_train_reduced['data']]
# cleaned_test_corpus = [clean_document(doc) for doc in newsgroups_test_reduced['data']]
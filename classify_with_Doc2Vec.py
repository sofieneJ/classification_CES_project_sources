from tools import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


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


def classify():
  # my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
  # X_train, y_train = generate_data_set(my_cats, subset='train')
  # X_test, y_test = generate_data_set(my_cats, subset='test')

  X_train = np.loadtxt('doc2vec_data\\X_train.csv', delimiter=',')
  y_train = np.loadtxt('doc2vec_data\\y_train.csv', delimiter=',')
  X_test = np.loadtxt('doc2vec_data\\X_test.csv', delimiter=',')
  y_test = np.loadtxt('doc2vec_data\\y_test.csv', delimiter=',')

  scaler = StandardScaler()
  # scaler.fit(X_train)
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  clf = knn(n_neighbors=5, weights='distance', metric='cosine')
  clf.fit(X_train, y_train)
  print ('{} with Doc2Vec score {}'.format(clf.__class__.__name__, 
  clf.score(X_test, y_test)))
  clf = LogisticRegression()
  clf.fit(X_train, y_train)
  print ('LogisticRegression with Doc2Vec score %f' % clf.score(X_test, y_test))
  clf = SVC()
  clf.fit(X_train, y_train)
  print ('SVC with Doc2Vec score %f' % clf.score(X_test,y_test))
  clf = MLPClassifier(hidden_layer_sizes=(100,))
  clf.fit(X_train, y_train)
  print ('MLPClassifier with Doc2Vec score %f' % clf.score(X_test,y_test))


  ############# Precision adn Recall ##################
  print (' Now analyzing the precision/recall...')
  Y_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4])
  Y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
  n_classes = Y_train.shape[1]

  # Run classifier
  classifier = OneVsRestClassifier(SVC())
  classifier.fit(X_train, Y_train)
  y_score = classifier.decision_function(X_test)

  # For each class
  precision = dict()
  recall = dict()
  average_precision = dict()
  for i in range(n_classes):
      precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                          y_score[:, i])
      # print ('precision for class {} is {}'.format(i, precision[i]))
      # print ('recall for class {} is {}'.format(i, recall[i]))
      average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
      print ('average_precision for class {} is {}'.format(i, average_precision[i]))
  
  print ('precision length for class {} is {}'.format(0, len(precision[0])))


def generate_data_set(test_cats, subset='train'):
    data_set_list = [list(read_corpus(fetch_20newsgroups(subset=subset,
                                                            remove=('headers', 'footers', 'quotes'),
                                                            categories=[cat])['data'], 
                                        tokens_only=True, bRemoveStopWords = True, bFastText=False))\
                        for cat in test_cats]

    corpus = [doc for categroy_list in data_set_list for doc in categroy_list ]

    # print ('raw corpus size : {}'.format(len(corpus)))
    categories_lengths=[len(cat_liste) for cat_liste in data_set_list]
    categories = [[k for _ in range(0,length)] for k,length in enumerate(categories_lengths)]
    cats = [cat for elem_list in categories for cat in elem_list]  
    y = np.array(cats)
    
    
    model_path = "model\\my_doc2vec_20news_model"
    model = gensim.models.doc2vec.Doc2Vec.load(model_path)
    t0 = time()
    X= np.vstack([model.infer_vector(doc) for doc in corpus])
    print (f'it took {time() - t0} s to generate doc2vec')

    np.savetxt('doc2vec_data\\X_{}.csv'.format(subset), X, delimiter=",")
    np.savetxt('doc2vec_data\\y_{}.csv'.format(subset), y, delimiter=",")

    # category_size = min ([len(data_set_list[k]) for k, _ in enumerate(data_set_list)])
    # X = np.vstack(np.array(model.infer_vector(data_set_list[i][k])) \
    #     for i, _ in enumerate(test_cats) for k in range(0, category_size) )
    # tags = [i for i, _ in enumerate(test_cats) for k in range(0, category_size) ]
    # y = np.array(tags)

    print ('subset {} shape {}:'.format(subset,X.shape))
    # print (y.shape)

    return X, y

def plotDoc2VecSimilarity ():
  my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
  X_train, _ = generate_data_set(my_cats, subset='train')
  scaler = StandardScaler()
  # scaler = MinMaxScaler()
  X_train = scaler.fit_transform(X_train)
  similarity_matrix = cosine_similarity(X_train)

#   plt.figure()
  plt.title('Cosine similarity of Doc2Vec representation')
  plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
  plt.show()

def analyse_accuracy():
  X_train = np.loadtxt('doc2vec_data\\X_train.csv', delimiter=',')
  y_train = np.loadtxt('doc2vec_data\\y_train.csv', delimiter=',')
  X_test = np.loadtxt('doc2vec_data\\X_test.csv', delimiter=',')
  y_test = np.loadtxt('doc2vec_data\\y_test.csv', delimiter=',')

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
  print (f'{clf.__class__.__name__} with doc2vec score {clf.score(X_test, y_test)}')

  my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
  print (classification_report(y_test,y_pred,target_names = my_cats))

if __name__ == "__main__":
  my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
  # generate_data_set(test_cats = my_cats, subset='train')
  # generate_data_set(test_cats = my_cats, subset='test')
  # classify()
  plotDoc2VecSimilarity()
  # draw_tsne('doc2vec_data\\X_train.csv', 'doc2vec_data\\y_train.csv', method='doc2vec')
  # analyse_accuracy()


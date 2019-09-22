from tools import *
from elmoformanylangs import Embedder

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import os

# with open("..\\data\\training_corpus.txt", 'r') as myfile:
#     print (myfile.readline())
# e = Embedder('..\\PreTrainedELMO_FR')

my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']


def classify():
  X_train = np.vstack([np.loadtxt('ELmo_20news_group_rep\\X_3L_train{}.csv'.format(i), delimiter=',') 
    for i in range(1,4)])
  y_train = np.loadtxt('ELmo_20news_group_rep\\y_train.csv', delimiter=',')
  X_test = np.vstack([np.loadtxt('ELmo_20news_group_rep\\X_3L_test{}.csv'.format(i), delimiter=',') 
    for i in range(1,4)])
  y_test = np.loadtxt('ELmo_20news_group_rep\\y_test.csv', delimiter=',')
  # train_index = np.where(X_train.any(axis=1))[0]
  # test_index = np.where(X_test.any(axis=1))[0]

  print (X_train.shape)
  print (y_train.shape)
  print (X_test.shape)
  print (y_test.shape)

  #################################   SHUFFLE   ##########################################""""
  index = np.arange(0,X_train.shape[0])
  np.random.shuffle(index)
  X_train = X_train[index]
  y_train = y_train[index]

  #################################   SCLALE   ##########################################""""
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  clf = knn()
  clf.fit(X_train, y_train)
  print (f'{clf.__class__.__name__} with ELMo reps score {clf.score(X_test, y_test)}' )
  clf = LogisticRegression()
  clf.fit(X_train, y_train)
  print ('LogisticRegression with ELMo reps score %f' % clf.score(X_test, y_test))
  t0 = time()
  clf = MLPClassifier(hidden_layer_sizes=(2000,2000), verbose=False)
  clf.fit(X_train, y_train)
  print (f'MLP took {time() - t0} to train')
  print ('MLPClassifier with ELMo reps score %f' % clf.score(X_test,y_test))
  clf = SVC()
  clf.fit(X_train, y_train)
  print ('rbf SVM with ELMo reps score %f' % clf.score(X_test,y_test))

def analyse_accuracy():
  X_train = np.vstack(np.loadtxt('ELmo_20news_group_rep\\X_3L_train{}.csv'.format(i), delimiter=',') 
    for i in range(1,4))
  y_train = np.loadtxt('ELmo_20news_group_rep\\y_train.csv', delimiter=',')
  X_test = np.vstack(np.loadtxt('ELmo_20news_group_rep\\X_3L_test{}.csv'.format(i), delimiter=',') 
    for i in range(1,4))
  y_test = np.loadtxt('ELmo_20news_group_rep\\y_test.csv', delimiter=',')

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
  print (f'{clf.__class__.__name__} with FastText score {clf.score(X_test, y_test)}')

  my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
  print (classification_report(y_test,y_pred,target_names = my_cats))



def BuildDataSet(subset='train', output = -1):
  dataset_list = [list(read_corpus(fetch_20newsgroups(subset=subset,
                                          remove=('headers', 'footers', 'quotes'),
                                    categories=[cat])['data'],
                                    tokens_only=True,  bFastText=False, bRemoveStopWords=True))\
                        for cat in my_cats]
  corpus = [doc for categroy_list in dataset_list for doc in categroy_list ]

  categories_lengths=[len(cat_liste) for cat_liste in dataset_list]
  categories = [[k for _ in range(0,length)] for k,length in enumerate(categories_lengths)]
  cats = [cat for elem_list in categories for cat in elem_list]  
  y = np.array(cats)
  np.savetxt('ELmo_20news_group_rep\\y_{}.csv'.format(subset), y, delimiter=",")
  
  print ('raw corpus ELMO Rep {} size {}'.format(subset, len(corpus)))
  e = Embedder('..\\PreTrainedElmo_EN', batch_size=64)
  
  try:
    os.remove('ELmo_20news_group_rep\\X_3L_{}1.csv'.format(subset))
  except:
    pass
  t0 = time()
  np_em= np.vstack([np.mean(np.array(e.sents2elmo([corpus[i]], output_layer=-2)[0]), axis=1).reshape(-1) 
      for i in range(0,len(corpus)//3)])
  np.savetxt('ELmo_20news_group_rep\\X_3L_{}1.csv'.format(subset), np_em, delimiter=',')
  print (f'finished generated 1st chunk of elmo reps in {time() - t0} seconds')


  try:
    os.remove('ELmo_20news_group_rep\\X_3L_{}2.csv'.format(subset))
  except:
    pass
  t0 = time()
  np_em= np.vstack([np.mean(np.array(e.sents2elmo([corpus[i]], output_layer=-2)[0]), axis=1).reshape(-1) 
      for i in range(len(corpus)//3, 2*(len(corpus)//3))])
  np.savetxt('ELmo_20news_group_rep\\X_3L_{}2.csv'.format(subset), np_em, delimiter=',')
  print (f'finished generated 2nd chunk of elmo reps in {time() - t0} seconds')


  try:
    os.remove('ELmo_20news_group_rep\\X_3L_{}3.csv'.format(subset))
  except:
    pass
  t0 = time()
  np_em= np.vstack([np.mean(np.array(e.sents2elmo([corpus[i]], output_layer=-2)[0]), axis=1).reshape(-1) 
      for i in range(2*(len(corpus)//3),len(corpus))])
  np.savetxt('ELmo_20news_group_rep\\X_3L_{}3.csv'.format(subset), np_em, delimiter=',')
  print (f'finished generated 2nd chunk of elmo reps in {time() - t0} seconds')
  

def plotElmOTextSimilarity ():
    X_train = np.vstack(np.loadtxt('ELmo_20news_group_rep\\X_3L_train{}.csv'.format(i), delimiter=',') 
                        for i in range(1,4))
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    similarity_matrix = cosine_similarity(X_train, X_train)
    
    plt.figure()
    plt.title('Cosine similarity of ElMO representation')
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.show()

if __name__ == "__main__":  
    # BuildDataSet(subset='train')
    # BuildDataSet(subset='test')
    # classify()
    # analyse_accuracy()
    plotElmOTextSimilarity()

    ##################T SNE############################""
    # X_train = np.vstack([np.loadtxt('ELmo_20news_group_rep\\X_train{}.csv'.format(i), delimiter=',') 
    #                     for i in range(1,4)])
    # y_train = np.loadtxt('ELmo_20news_group_rep\\y_train.csv', delimiter=',') 
    # draw_tsne_with_datasets(X=X_train, y=y_train, method='ElMO')
    
    
    # e = Embedder('..\\PreTrainedElmo_EN', batch_size=64)
    # Elmo_Embeddings = e.sents2elmo([['most', 'boys', 'like', 'big', 'cats']], output_layer=-2)
    # np_em= np.mean(np.array(Elmo_Embeddings[0]), axis=1).reshape(-1)
    # print (np_em.shape)

    # np_token_ems = [np.mean(np.array(token_em), axis=1) for token_em in Elmo_Embeddings]
    # np_sent_em = (sum(np_token_ems)/len(np_token_ems)).reshape(-1)
    # print (np_sent_em.shape)
    # print (np.mean(np_ems[0], axis=1).shape)
    # print (np.mean())
    # print (np_em[0].shape)
    # print (np_em[0])

# newsgroups_train = fetch_20newsgroups(subset='train',
#                                     categories=my_cats,
#                                   remove=('headers', 'footers', 'quotes'))
# train_corpus = list(read_corpus(newsgroups_train['data'], tokens_only=True,
#                                 bRemoveStopWords=True, bFastText=False))

# sents: the list of lists which store the sentences after segment if necessary.
# output_layer: the target layer to output.
# 0 for the word encoder
# 1 for the first LSTM hidden layer
# 2 for the second LSTM hidden layer
# -1 for an average of 3 layers. (default)
# -2 for all 3 layers
# batch_size = 1
# e = Embedder('..\\PreTrainedElmo_EN', batch_size=64)
# Elmo_Embeddings = e.sents2elmo(train_corpus[0:40], output_layer=-1)
# print('batch length {}'.format(len(train_corpus[0:2])))
# print ('Elmo output length {}'.format(len(Elmo_Embeddings)))
# print('Elmo_Embeddings[0].shape {}'.format(Elmo_Embeddings[0].shape))
# print('Elmo_Embeddings[1].shape {}'.format(Elmo_Embeddings[1].shape))
# print(len(Elmo_Embeddings[0][0]))
# print(Elmo_Embeddings[0])
# for i in range(0,len(train_corpus),40):
#   print ((i, min(i+40, len(train_corpus))))
#   Elmo_batch_Embeddings = e.sents2elmo(train_corpus[i:min(i+40, len(train_corpus))], output_layer=-1)
# Elmo_Embeddings = [e.sents2elmo(train_corpus[i:min(i+batch_size, len(train_corpus))], output_layer=-1) for i in range(0,len(train_corpus),batch_size)]
# Elmo_Embeddings = [e.sents2elmo([train_corpus[i]]) for i in range(0,len(train_corpus))]
# X = np.vstack(np.mean(em[0],axis=0) for em in Elmo_Embeddings)
# print ('X.shape {}'.format(X.shape))
# np.savetxt('ELmo_20news_group_rep\\X_{}.csv'.format('train'), X, delimiter=",")

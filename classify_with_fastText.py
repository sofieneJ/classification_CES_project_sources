from tools import *

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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



def generate_data_set(test_cats, subset='train', dim=100):

    data_set_list = [list(read_corpus(fetch_20newsgroups(subset=subset,
                                                            remove=('headers', 'footers', 'quotes'),
                                                            categories=[cat])['data'],
                                           tokens_only=True, bRemoveStopWords=False, bFastText=True))\
                        for cat in test_cats]

    corpus = [doc for categroy_list in data_set_list for doc in categroy_list ]

    # print ('raw corpus size : {}'.format(len(corpus)))
    categories_lengths=[len(cat_liste) for cat_liste in data_set_list]
    categories = [[k for _ in range(0,length)] for k,length in enumerate(categories_lengths)]
    cats = [cat for elem_list in categories for cat in elem_list]  
    y = np.array(cats)


    model_path = "model\\myFT20newsGroupModel_skipgram_{}.bin".format(dim)
    model = ft.load_model(model_path)
    t0 = time()
    X = np.vstack([model.get_sentence_vector(doc) for doc in corpus])
    print(f'it took {time()- t0} seconds to infer fasttext embeddings')

    print (f'subset {subset} shape {X.shape}:')

    np.savetxt('fastText_data\\X{}_{}.csv'.format(dim, subset), X, delimiter=",")
    np.savetxt('fastText_data\\y{}_{}.csv'.format(dim, subset), y, delimiter=",")

    return X, y

def plotFastTextSimilarity ():
  my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
  X_train, _ = generate_data_set(my_cats, subset='train')
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  similarity_matrix = np.inner(X_train, X_train)

  plt.figure()
  plt.title('Cosine similarity of FastText representation')
  plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
  plt.show()


def classify(dim):
  # my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
  # X_train, y_train = generate_data_set(my_cats, subset='train')
  # X_test, y_test = generate_data_set(my_cats, subset='test')
  X_train = np.loadtxt('fastText_data\\X{}_train.csv'.format(dim), delimiter=',')
  y_train = np.loadtxt('fastText_data\\y{}_train.csv'.format(dim), delimiter=',')
  X_test = np.loadtxt('fastText_data\\X{}_test.csv'.format(dim), delimiter=',')
  y_test = np.loadtxt('fastText_data\\y{}_test.csv'.format(dim), delimiter=',')

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  clf = knn()
  clf.fit(X_train, y_train)
  print (f'{clf.__class__.__name__} with FastText score {clf.score(X_test, y_test)}')

  clf = LogisticRegression()
  clf.fit(X_train, y_train)
  print (f'{clf.__class__.__name__} with FastText score {clf.score(X_test, y_test)}')
  clf = MLPClassifier(hidden_layer_sizes=(1000,1000))
  clf.fit(X_train, y_train)
  print (f'{clf.__class__.__name__} with FastText score {clf.score(X_test, y_test)}')
  clf = SVC()
  clf.fit(X_train, y_train)
  print (f'{clf.__class__.__name__} with FastText score {clf.score(X_test, y_test)}')

def compute_f1_score(dim=100):
  X_train = np.loadtxt('fastText_data\\X{}_train.csv'.format(dim), delimiter=',')
  y_train = np.loadtxt('fastText_data\\y{}_train.csv'.format(dim), delimiter=',')
  X_test = np.loadtxt('fastText_data\\X{}_test.csv'.format(dim), delimiter=',')
  y_test = np.loadtxt('fastText_data\\y{}_test.csv'.format(dim), delimiter=',')

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

  print (f'f1 score of {clf.__class__.__name__} is {f1_score(y_test, y_pred, average=None)}')


if __name__ == "__main__":  
  my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
  embedding_size = 100
  generate_data_set(test_cats = my_cats, subset='train', dim=embedding_size)
  generate_data_set(test_cats = my_cats, subset='test', dim=embedding_size)
  classify(dim=embedding_size)

  # plotFastTextSimilarity()
  # draw_tsne('fastText_data\\X_train.csv', 'fastText_data\\y_train.csv', method='fastText')
  compute_f1_score()


import tensorflow as tf
import tensorflow_hub as hub
from tools import *


from tools import *

import os
import re





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
    np.savetxt(fname='local_universal_encoder_data\\y_{}.csv'.format(subset), X=y, delimiter=',')

    t0 = time()
    embed = hub.Module("C:\\Users\\sofiene.jenzri\\Documents\\OneDrive - UiPath\\Documents\\DataScience\\project\\sources\\UniversalEncoderGoogle\\large_3")
    # Compute a representation for each message, showing various lengths supported.
    # word = "Elephant"
    # sentence = "I am a sentence for which I would like to get its embedding."
    # paragraph = (
    #     "Universal Sentence Encoder embeddings also support short paragraphs. "
    #     "There is no hard limit on how long the paragraph is. Roughly, the longer "
    #     "the more 'diluted' the embedding will be.")
    # messages = [word, sentence, paragraph]
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        doc_embeddings = session.run(embed(corpus))
        X = np.vstack(doc_embeddings)
    print (f'{subset} embeddings generated in {time() - t0} seconds')

    # t0 = time()
    # #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
    # module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" 
    # embed = hub.Module(module_url)
    # tf.logging.set_verbosity(tf.logging.ERROR)

    # with tf.Session() as session:
    #     session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    #     message_embeddings = session.run(embed(corpus))
    #     X = np.vstack(message_embeddings)
    # print (f'embedding generated in {time() - t0} seconds')


    print ('subset {} shape {}:'.format(subset,X.shape))
    np.savetxt(fname='local_universal_encoder_data\\X_{}.csv'.format(subset), X=X, delimiter=',')

    return X, y

def plotBERTTextSimilarity ():
    X_train = np.loadtxt('local_universal_encoder_data\\X_train.csv', delimiter=',')
    # y_train = np.loadtxt('local_universal_encoder_data\\y_train.csv', delimiter=',')

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
    X_train = np.loadtxt('local_universal_encoder_data\\X_train.csv', delimiter=',')
    y_train = np.loadtxt('local_universal_encoder_data\\y_train.csv', delimiter=',')
    X_test = np.loadtxt('local_universal_encoder_data\\X_test.csv', delimiter=',')
    y_test = np.loadtxt('local_universal_encoder_data\\y_test.csv', delimiter=',')

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
    X_train = np.loadtxt('local_universal_encoder_data\\X_train.csv', delimiter=',')
    y_train = np.loadtxt('local_universal_encoder_data\\y_train.csv', delimiter=',')
    X_test = np.loadtxt('local_universal_encoder_data\\X_test.csv', delimiter=',')
    y_test = np.loadtxt('local_universal_encoder_data\\y_test.csv', delimiter=',')

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
    print (f'{clf.__class__.__name__} with local universal encoder score {clf.score(X_test, y_test)}')

    print (f'f1 score of {clf.__class__.__name__} is {f1_score(y_test, y_pred, average=None)}')


if __name__ == "__main__":  
    my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
    X_train, y_train = generate_data_set(my_cats, subset='train')
    X_test, y_test = generate_data_set(my_cats, subset='test')
    # classify()
    # plotBERTTextSimilarity()
    compute_f1_score()
    # draw_tsne('local_universal_encoder_data\\X_train.csv', 'local_universal_encoder_data\\y_train.csv', method='BERT')


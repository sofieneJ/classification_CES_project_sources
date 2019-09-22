import numpy as np
import pandas as pd
import fastText as ft
from sklearn.datasets import fetch_20newsgroups
import gensim
import nltk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.offsetbox import AnchoredText
from matplotlib.offsetbox import AnnotationBbox


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn import manifold, decomposition

from time import time
import sklearn.metrics as met
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import  SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity


english_words = set(nltk.corpus.words.words())
english_words.remove('ax')


def clean_document (doc, bFastText, bRemoveStopWords, min_token_len = 2, max_token_len=25):
    document =doc
    if bRemoveStopWords:
        document = gensim.parsing.remove_stopwords(document)

    if bFastText:
        return ' '.join([w for w in gensim.utils.simple_preprocess(document, min_len=min_token_len, max_len=max_token_len) if w in english_words])
    else:
        return [w for w in gensim.utils.simple_preprocess(document, min_len=min_token_len, max_len=max_token_len) if w in english_words]
        # return gensim.utils.simple_preprocess(document)


def my_cosine_similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2



def read_corpus(corpus, tokens_only, bFastText, bRemoveStopWords):
    # with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
    for i, doc in enumerate(corpus):
        if tokens_only:
            ret_doc = clean_document(doc, bFastText, bRemoveStopWords)
            len_document = len (ret_doc.split()) if bFastText else len (ret_doc)
            if len_document >3:
                yield ret_doc
        else:
            # For training data, add tags
            if bRemoveStopWords:
                ret_doc =  gensim.utils.simple_preprocess(gensim.parsing.remove_stopwords(doc))
            else:
                ret_doc = gensim.utils.simple_preprocess(doc)
            if len (ret_doc) > 3:
                yield gensim.models.doc2vec.TaggedDocument(ret_doc, [i])




def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt




# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(int (y[i])),
                 color=plt.cm.Set1((y[i]+1) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
    # shown_labels = np.array([[1., 1.]])  # just something big
    # for i in range(X.shape[0]):
    #     dist = np.sum((X[i] - shown_labels) ** 2, 1)
    #     if np.min(dist) < 0.005:
    #         # don't show points that are too close
    #         continue
    #     shown_labels = np.r_[shown_labels, [X[i]]]
    #     label_box = AnnotationBbox( AnchoredText(classes_dict[y[i]], loc='lower right'), X[i])
    #     ax.add_artist(label_box)
    
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

def draw_tsne(X_file_path, y_file_path, method):

    X = np.loadtxt(X_file_path, delimiter=',')
    y = np.loadtxt(y_file_path, delimiter=',')
    
    index = np.arange(0,X.shape[0])
    np.random.shuffle(index)    
    X = X[index]
    y = y[index]

    # ----------------------------------------------------------------------
    # t-SNE embedding of the digits dataset

    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X = X_tsne, y= y,
        title="t-SNE embedding of the documents using {} (time {})".format(method, time() - t0))

def draw_tsne_with_datasets(X, y, method):

    # X = np.loadtxt(X_file_path, delimiter=',')
    # y = np.loadtxt(y_file_path, delimiter=',')
    
    index = np.arange(0,X.shape[0])
    np.random.shuffle(index)    
    X = X[index]
    y = y[index]

    # ----------------------------------------------------------------------
    # t-SNE embedding of the digits dataset

    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X = X_tsne, y= y,
        title="t-SNE embedding of the documents using {} (time {})".format(method, time() - t0))


def initialize_centroides (X, K):
    feat_dim = X.shape[1]
    centroids = np.zeros(shape=(K, feat_dim))
    for i in range (0, K):
        centroids[i] = np.quantile(X, (2*i+1)/(2*K), axis=0)

    return centroids


def assign_data(X, _centroids):
    dist = met.pairwise_distances(X, _centroids, metric='cosine')
    classes = np.argmin(dist, axis=1)
    return classes


def compute_centroides (X, K, classes):
    feat_dim = X.shape[1]
    centroids = np.zeros(shape=(K, feat_dim))
    unique = np.unique(classes)
    # index_0 = classes == unique[0]
    for i, classe in enumerate(unique):
        cluster = X[classes == classe]
        # print (cluster.shape)
        centroids[i] = np.mean(cluster,axis=0)
    return centroids

def run_consine_cluster (X, K, bVerbose=False):
    print('starting cosine K-means')
    centroids = initialize_centroides(X, K)
    max_iter = 100
    iter=0
    centroids_shift = 1
    while centroids_shift > 0 and iter<max_iter:
        classes = assign_data(X, centroids)
        new_centroids = compute_centroides (X, K, classes)
        centroids_shift = np.linalg.norm(centroids-new_centroids)
        if bVerbose:
            print (' iter % d with centroids distance = %f' %(iter, centroids_shift) )
        iter = iter+1
        centroids = new_centroids

    return classes

# if __name__=='__main__':
#     draw_tsne('fastText_data\\X100_train.csv', 'fastText_data\\y100_train.csv', method='Fasttext')
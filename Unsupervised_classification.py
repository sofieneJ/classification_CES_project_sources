from tools import *
from sklearn.decomposition import PCA
from pylab import *

def classify(method = 'BERT', bPCA = True, nb_clusters=5):

    # my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
    # my_cats = ['rec.autos', 'soc.religion.christian']
    # X_train, y_train = generate_data_set(my_cats, subset='train')

   

    if method == 'BERT':
        X_train = np.loadtxt('bert_embeddings_data\\X_train.csv', delimiter=',')
        if bPCA:
            pca = PCA(n_components=200)
            X_train = pca.fit_transform(X_train)
        # y_train = np.loadtxt('bert_embeddings_data\\y_train.csv', delimiter=',')


    if method == 'ElMO':
        X_train = np.vstack([np.loadtxt('ELmo_20news_group_rep\\X_train{}.csv'.format(i), delimiter=',') 
            for i in range(1,4)])
        if bPCA:
            pca = PCA(n_components=200)
            X_train = pca.fit_transform(X_train)
        

    if method == 'FastText':
        X_train = np.loadtxt('fastText_data\\X100_train.csv', delimiter=',')
        if bPCA:
            pca = PCA(n_components=100)
            X_train = pca.fit_transform(X_train)
    
    if method == 'doc2vec':
        X_train = np.loadtxt('doc2vec_data\\X_train.csv', delimiter=',')
        if bPCA:
            pca = PCA(n_components=100)
            X_train = pca.fit_transform(X_train)

    if method == 'all':
        subplot(2,2,1)
        X_train = np.loadtxt('bert_embeddings_data\\X_train.csv', delimiter=',')
        if bPCA:
            pca = PCA(n_components=200)
            X_train = pca.fit_transform(X_train)
        classes = run_consine_cluster (X=X_train, K=nb_clusters)
        title('BERT')
        plot(classes, '.')

        subplot(2,2,2)
        X_train = np.vstack([np.loadtxt('ELmo_20news_group_rep\\X_3L_train{}.csv'.format(i), delimiter=',') 
            for i in range(1,4)])
        if bPCA:
            pca = PCA(n_components=200)
            X_train = pca.fit_transform(X_train)
        classes = run_consine_cluster (X=X_train, K=nb_clusters)
        title('ElMo')
        plot(classes, '.')

        subplot(2,2,3)
        X_train = np.loadtxt('fastText_data\\X100_train.csv', delimiter=',')
        if bPCA:
            pca = PCA(n_components=100)
            X_train = pca.fit_transform(X_train)
        classes = run_consine_cluster (X=X_train, K=nb_clusters)
        title('FastText')
        plot(classes, '.')

        subplot(2,2,4)
        X_train = np.loadtxt('custom_doc2vec_data\\X_train.csv', delimiter=',')
        if bPCA:
            pca = PCA(n_components=100)
            X_train = pca.fit_transform(X_train)
        classes = run_consine_cluster (X=X_train, K=nb_clusters)
        title('word2vec + TF-IDF')
        plot(classes, '.')
    else:
        classes = run_consine_cluster (X=X_train, K=nb_clusters)
        plt.figure()
        plt.plot (classes, '.')
        plt.title('clustering with %s' %method)


def retrieve_data (i):
    if i==0: 
        method = 'BERT'
        X_train = np.loadtxt('bert_embeddings_data\\X_train.csv', delimiter=',')
        if bPCA:
            pca = PCA(n_components=200)
            X_train = pca.fit_transform(X_train)
        # y_train = np.loadtxt('bert_embeddings_data\\y_train.csv', delimiter=',')


    if i==1: 
        method = 'ElMO'
        X_train = np.vstack([np.loadtxt('ELmo_20news_group_rep\\X_train{}.csv'.format(i), delimiter=',') 
            for i in range(1,4)])
        if bPCA:
            pca = PCA(n_components=200)
            X_train = pca.fit_transform(X_train)
        
    if i==2: 
        method =  'FastText'
        X_train = np.loadtxt('fastText_data\\X100_train.csv', delimiter=',')
        if bPCA:
            pca = PCA(n_components=100)
            X_train = pca.fit_transform(X_train)
    
    if i==3: 
        method =  'word2vec+TF-IDF'
        X_train = np.loadtxt('custom_doc2vec_data\\X_train.csv', delimiter=',')
        if bPCA:
            pca = PCA(n_components=100)
            X_train = pca.fit_transform(X_train)

    return X_train, method

def evaluate_with_silouhaite(method='BERT', nb_clusters=5):

    fig, ax = plt.subplots(2, 2)
    ax= ax.reshape(-1)

    for i_ax in range(0,4):
        X_train, method = retrieve_data(i_ax)
        # pca = PCA(n_components=200)
        # X_train = pca.fit_transform(X_train)
        cluster_labels = run_consine_cluster (X=X_train, K=nb_clusters)

        # Create a subplot with 1 row and 2 columns
        # fig, ax[i_ax] = plt.subplots(1, 1)

        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax[i_ax].set_xlim([-0.5, 0.5])
        # The (5+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax[i_ax].set_ylim([0, len(X_train) + (nb_clusters + 1) * 10])


        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X_train, cluster_labels, metric='cosine')
        print("For nb_clusters =", nb_clusters, f' using {method} ',
            "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X_train, cluster_labels)

        y_lower = 10
        for i in range(nb_clusters):

            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / nb_clusters)
            ax[i_ax].fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax[i_ax].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax[i_ax].set_title("The silhouette plot using {} embeddings.".format(method))
        ax[i_ax].set_xlabel("The silhouette coefficient values")
        ax[i_ax].set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax[i_ax].axvline(x=silhouette_avg, color="red", linestyle="--")

        ax[i_ax].set_yticks([])  # Clear the yaxis labels / ticks
        ax[i_ax].set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])

        plt.suptitle((f'Silhouette analysis for KMeans clustering with K={nb_clusters}'),
                    fontsize=14, fontweight='bold')
        

if __name__ == "__main__":  
    # train_word2vec()
    # train_TFIDF()
    # BuildDataSet(subset='train')
    # BuildDataSet(subset='test')
    bPCA = True
    # classify(method='FastText',  bPCA=bPCA)
    # classify(method='BERT', bPCA=bPCA)
    # classify(method='ElMO', bPCA=bPCA)
    # classify(method='doc2vec', bPCA=bPCA)
    # plotCustomRepSimilarity()
    K= 3
    classify(method='all',nb_clusters=K)
    evaluate_with_silouhaite(nb_clusters=K)

    # fig, ax = plt.subplots(2, 2)
    # ax.reshape(-1)
    # for i in range(0,4):
    #     ax[i].set_title(f'hello{i}')
    
    plt.show()
from tools import *
# from elmoformanylangs import Embedder
from pandas import DataFrame as df

# sentence = 'from article colorado colorado eric let like this the similarity liter engine displacement actually the coupe the funky looking new sedan share liter six es popular small sedan the luxury sports coupe new luxury sedan es base executive sedan all look completely different'
# # sentence = gensim.parsing.preprocessing.stem_text(sentence) 
# print(sentence)
# sentence = clean_document(sentence, bFastText = True, bRemoveStopWords = True)

# print(sentence)

# vec = np.zeros(shape=(2,3))
# vec2 = np.ones(shape=(2,3))
# print (mean([vec, vec2])/len([vec, vec2]))
# print ( 'aardvark' in english_words)

# dico = {
#     'a':1,
#     'b':2
# }

# try:
#     print (dico['a'])
#     print (dico['c'])
# except:
#     print('inference problem')


# model_path = "model\\my_word2vec_20news_model"
# w2v_model = gensim.models.word2vec.Word2Vec.load(model_path)
# print (w2v_model.wv['emptiness'])

# sentence = 'from article colorado colorado eric let like this the similarity liter engine displacement actually the coupe the funky looking new sedan share liter six es popular small sedan the luxury sports coupe new luxury sedan es base executive sedan all look completely different'
# print (gensim.utils.simple_preprocess(sentence))
# def myCodition (x):
#     return x<5
# my_cond = lambda x: x<5
# x = np.arange(1,10)
# print (x[np.where(my_cond(x))[0]])


#############################ELMO#################################""""
# batch_size = 1
# e = Embedder('..\\PreTrainedElmo_EN', batch_size=64)
# # Elmo_Embeddings = e.sents2elmo([['love']], output_layer=-1)
# my_cats = ['sci.med']
# newsgroups_train = fetch_20newsgroups(subset='train',
#                                     categories=my_cats,
#                                   remove=('headers', 'footers', 'quotes'))
# train_corpus = list(read_corpus(newsgroups_train['data'], tokens_only=True,
#                                 bRemoveStopWords=True, bFastText=False))
# train_corpus = sorted(train_corpus,key=len,reverse=False)
# index = 100
# print (len(train_corpus[index]))

# Elmo_Embeddings = e.sents2elmo([train_corpus[index]], output_layer=0)
# print(len(Elmo_Embeddings))
# print(Elmo_Embeddings[0].shape)
# print(Elmo_Embeddings[0][0,0:10])
# print(Elmo_Embeddings[0][0,0,0:10])
# print(Elmo_Embeddings[0][0,1,0:10])
# print(Elmo_Embeddings[0][1,0,0:10])
# print(Elmo_Embeddings[0][2,0,0:10])
# print (np.mean(Elmo_Embeddings,axis=0).shape)

# with open('test.csv', mode='a') as myFile:
#     myFile.write('{}'.format(Elmo_Embeddings[0].tolist()).strip('[').strip(']').replace(' ',''))
#     myFile.write('\n')

# model_path = "model\\my_word2vec_20news_model"
# w2v_model = gensim.models.word2vec.Word2Vec.load(model_path)

# print (train_corpus[23:30])
# centroids = [['car', 'engine', 'drive', 'speed'],
# ['religion', 'jesus', 'god', 'believe', 'heaven', 'sin'],
# ['baseball', 'player', 'run', 'sport', 'hit', 'bat', 'rotation'],
# ['electronics', 'conductive', 'power', 'resistor', 'circuit'],
# ['medical', 'methodology', 'science', 'molecule', 'virus']]


###############################################Tensorflow#########################################################
# import tensorflow as tf
# mnist = tf.keras.datasets.mnist

# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)

# cats = ['alt.atheism',
#  'comp.graphics',
#  'comp.os.ms-windows.misc',
#  'comp.sys.ibm.pc.hardware',
#  'comp.sys.mac.hardware',
#  'comp.windows.x',
#  'misc.forsale',
#  'rec.autos',
#  'rec.motorcycles',
#  'rec.sport.baseball',
#  'rec.sport.hockey',
#  'sci.crypt',
#  'sci.electronics',
#  'sci.med',
#  'sci.space',
#  'soc.religion.christian',
#  'talk.politics.guns',
#  'talk.politics.mideast',
#  'talk.politics.misc',
#  'talk.religion.misc']

# my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
# data_set_list = [list(read_corpus(fetch_20newsgroups(subset='test',
#                                                         remove=('headers', 'footers', 'quotes'),
#                                                         categories=[cat])['data'],
#                                         tokens_only=True, bRemoveStopWords=True, bFastText=False))\
#                     for cat in my_cats]

# corpus = [doc for categroy_list in data_set_list for doc in categroy_list ]
# list_len_df = df(np.array([len(doc) for doc in corpus]))
# print (corpus[0])
# print (list_len_df.describe())
# print (list_len_df.sum())
# word_exist = [ lambda doc : 1 if 'cryptozoology' in doc else 0 for doc in corpus ]
# print (sum(word_exist))

# file_path = "C:\\Users\\sofiene.jenzri\\Documents\\OneDrive - UiPath\\Documents\\DataScience\\bert_models\\working\\train.tf_record"
# with open(file=file_path, mode='w') as myFile:
#     pass

 
## TENSOR FLOW HUB download
# import tensorflow_hub as hub
# m=hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")

# my_list = ['bob', 'alice', 'sebastien']
# np_list = np.array(my_list).reshape(len(my_list),1)
# print (np_list)
# np_ind = np.apply_along_axis(lambda x: str(x[0]) if len(x[0]) >3 else 'videeee', axis=1, arr=np_list)
# print (np_ind)
# np_ind = np_ind[np_ind != 'videeee']
# print(np_ind)

my_list = ['bob', 'alice', 'sebastien']
Ser_list = pd.Series(my_list)
print (Ser_list)
Ser_list = Ser_list.apply(lambda x: x if len(x)>3 else 'vide', convert_dtype=True)
print (Ser_list)
Ser_list = Ser_list[Ser_list != 'vide']
print(Ser_list)
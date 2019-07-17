from tools import *

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
newsgroups_train = fetch_20newsgroups(subset='train',
                                  remove=('headers', 'footers', 'quotes'))
                                  

train_corpus = list(read_corpus(newsgroups_train['data'], tokens_only=False,
                                bRemoveStopWords=True, bFastText=False))

model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=40)
# Build a Vocabulary
model.build_vocab(train_corpus)

# Time to Train
t0 = time()
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
print ('training of doc2ec done in {}'.format (time() - t0))
#save model
# fname = get_tmpfile("model\\my_doc2vec_model")
model_path = 'model\\my_doc2vec_20news_model'
model.save(model_path)

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



# print(len(newsgroups_train['data']))
# print (str(newsgroups_train['data'][2]).replace('\n',' ').replace('\t',' '))
# print (newsgroups_train['data'][1])

# nltk.download('words')
english_words = set(nltk.corpus.words.words())
english_words.remove('ax')

def dump_training_corpus (file_path):
    with open(file_path, mode='w') as myfile:
        for line in newsgroups_train['data']:
            clean_line = clean_document(line, bRemoveStopWords=False, bFastText=True) 
            myfile.write(clean_line)
            myfile.write('\n')

    myfile.close()


training_corpus_path='..\\..\\data_set\\20newsgroups\\training_corpus.txt'
dump_training_corpus(file_path=training_corpus_path)

# # # # # # TRAIN THE MODEL # # # # # # 
t0 = time()
model = ft.train_unsupervised(input=training_corpus_path, model='skipgram', verbose=0, dim=100)
print ('training of fastText done in {}'.format (time() - t0))

my_model_name= "model\\myFT20newsGroupModel_skipgram_100.bin"
# model.save_model(my_model_name)

model = ft.load_model(my_model_name)

# print (model.get_word_vector('machine'))
print (model.get_sentence_vector('I like milk and apples'))
# print (model.get_dimension())
print (model.get_subwords('cryptozoology')[0])
print ('cryptozoology embeeding', model.get_word_vector('cryptozoology'))
print ('ptozoo embedding ', model.get_word_vector('ptozoo'))
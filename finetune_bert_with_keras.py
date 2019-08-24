import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from bert.tokenization import FullTokenizer
# from tqdm import tqdm_notebook
from tensorflow.keras import backend as K
import gensim
import keras
from datetime import datetime

################################### cf https://github.com/strongio/keras-bert ##############################################################

# Initialize session
sess = tf.Session()


# Params for bert model and tokenization
g_bert_path = "C:\\Users\\sofiene.jenzri\\Documents\\OneDrive - UiPath\\Documents\\DataScience\\LM_models\\bert_models\\tf_hub_bert_uncased_L-12_H-768_A-12"
g_save_trained_model_path = 'model\\keras_fine_tuned_bert_classifier\\BertModel.h5'
max_seq_length = 256

from sklearn.datasets import fetch_20newsgroups
my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
label_list = [0, 1, 2, 3, 4]
DATA_COLUMN = 'text'
LABEL_COLUMN = 'cat'

def generate_raw_data_df(subset='train'):
    def my_simple_reading_func(corpus):
        for doc in corpus:
            # yield doc
            seq = gensim.utils.simple_preprocess(gensim.parsing.remove_stopwords(doc), min_len=2, max_len=25)
            yield ' '.join(seq)

    # my_simple_reading_func = lambda corpus :  yield doc for doc in corpus
    dataset_list = [list(my_simple_reading_func(fetch_20newsgroups(subset=subset,
                                          remove=('headers', 'footers', 'quotes'),
                                    categories=[cat])['data']))\
                        for cat in my_cats]

    categories_lengths=[len(cat_liste) for cat_liste in dataset_list]
    categories = [[k for _ in range(0,length)] for k,length in enumerate(categories_lengths)]
    cats_ser = pd.Series([cat for elem_list in categories for cat in elem_list])

    corpus_ser = pd.Series([doc for categroy_list in dataset_list for doc in categroy_list ])
    corpus_ser = corpus_ser.apply(lambda x: x if len(x.split())<max_seq_length and len(x.split())>10 else 'refused', convert_dtype=True) #220
    my_index = corpus_ser != 'refused'
    corpus_ser = corpus_ser[my_index].reindex()
    cats_ser =cats_ser[my_index].reindex()

    # print (corpus_ser.head())
    # print (cats_ser.head())

    doc_lens = np.array([len(doc.split()) for doc in corpus_ser])
    print (f'mean length is {np.mean(doc_lens)}')
    print (f'max length is {np.max(doc_lens)}')
    print ('raw corpus size : {}'.format(corpus_ser.size))


    data_dic = {DATA_COLUMN: corpus_ser, LABEL_COLUMN:cats_ser}
    ret_df = pd.DataFrame(data_dic)
    # print(ret_df.dtypes)
    print (ret_df.head())
    return ret_df
 
class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(g_bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    # for example in tqdm_notebook(examples, desc="Converting examples to features"):
    for example in examples:
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples

class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="first",
        bert_path=g_bert_path,
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

# Build model
def build_model(max_seq_length): 
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]
    
    bert_output = BertLayer(n_fine_tune_layers=1, pooling="mean")(bert_inputs) #pooling="first"
    dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    pred = tf.keras.layers.Dense(units=len(label_list), activation='softmax')(dense) #sigmoid
    
    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #binary_crossentropy
    model.summary()
    
    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

def train_classification(): 
    train_df = generate_raw_data_df(subset='train')
    test_df = generate_raw_data_df(subset='test')

    label_list = [0, 1, 2, 3, 4]
    # Create datasets (Only take up to max_seq_length words for memory)
    train_text = train_df[DATA_COLUMN].tolist()
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = train_df[LABEL_COLUMN].tolist()

    test_text = test_df[DATA_COLUMN].tolist()
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = test_df[LABEL_COLUMN].tolist()
    print (f' train text shape {train_text.shape}')
    print (f' test text shape {test_text.shape}')

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module()

    # Convert data to InputExample format
    train_examples = convert_text_to_examples(train_text, train_label)
    test_examples = convert_text_to_examples(test_text, test_label)

    # Convert to features
    (train_input_ids, train_input_masks, train_segment_ids, train_labels 
    ) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels
    ) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)


    model = build_model(max_seq_length)

    # Instantiate variables
    initialize_vars(sess)

    categorical_train_labels = keras.utils.to_categorical(train_labels, num_classes=len(label_list))
    categorical_test_labels = keras.utils.to_categorical(test_labels, num_classes=len(label_list))

    model.fit(
        [train_input_ids, train_input_masks, train_segment_ids], 
        categorical_train_labels,
        validation_data=([test_input_ids, test_input_masks, test_segment_ids], categorical_test_labels),
        epochs=1,
        batch_size=32
    )

    model.save(g_save_trained_model_path)
    pre_save_preds = model.predict([test_input_ids[0:100], 
                                    test_input_masks[0:100], 
                                    test_segment_ids[0:100]]
                                ) # predictions before we clear and reload model

    print (pre_save_preds)


def predict_with_finetuned_model():
    test_df = generate_raw_data_df(subset='test')
    test_text = test_df[DATA_COLUMN].tolist()
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = test_df[LABEL_COLUMN].tolist()

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module()

    test_examples = convert_text_to_examples(test_text, test_label)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels
    ) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)

    model = build_model(max_seq_length)
    initialize_vars(sess)
    model.load_weights(g_save_trained_model_path)

    # post_save_preds = model.predict([test_input_ids[0:2], 
    #                                 test_input_masks[0:2], 
    #                                 test_segment_ids[0:2]]
    #                             ) # predictions after we clear and reload model

    post_save_preds = model.predict([[test_input_ids[1000]], 
                                    [test_input_masks[1000]], 
                                    [test_segment_ids[1000]]]
                                ) # predictions after we clear and reload model

    # all(pre_save_preds == post_save_preds) # Are they the same?
    print (post_save_preds)


def predict_single_sample_with_finetuned_model(sample):
    seq = gensim.utils.simple_preprocess(gensim.parsing.remove_stopwords(sample), min_len=2, max_len=25)
    if (len(seq) > 256):
        raise NameError(
                f"sequence is too long: the obtained size after preprocessing is {len(seq)}. Maximum size is {max_seq_length}"
            )
    if (len(seq)<10):
        raise NameError(
                f"sequence is too short: the obtained size after preprocessing is {len(seq)}. Minimum size is 10"
            )
    
    bert_example = convert_text_to_examples(texts = [seq], labels=[-1])
    # bert_sample = InputExample(guid=None, text_a=" ".join(seq), text_b=None, label=-1)
    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module()

    input_id, input_mask, segment_id, _ = convert_single_example(
        tokenizer, bert_example[0], max_seq_length
    )

    model = build_model(max_seq_length)
    initialize_vars(sess)
    model.load_weights(g_save_trained_model_path)

    post_save_preds = model.predict([[input_id], 
                                    [input_mask], 
                                    [segment_id]]
                                ) # predictions after we clear and reload model

    # all(pre_save_preds == post_save_preds) # Are they the same?
    return post_save_preds[0]
    # print (post_save_preds)

if __name__=='__main__':
    # train_classification()
    # predict_with_finetuned_model()


    #########predict a single sample#######################
    my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
    cat = 'rec.sport.baseball'
    sample_index = 99
    sample = fetch_20newsgroups(subset='test',
                            remove=('headers', 'footers', 'quotes'),
                            categories=[cat])['data'][sample_index]
    print (sample)
    probs = predict_single_sample_with_finetuned_model(sample)
    prob_dict = {a[0]:a[1] for a in zip(my_cats, probs)}
    print (prob_dict)

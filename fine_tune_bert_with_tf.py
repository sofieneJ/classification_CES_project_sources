##### cf https://github.com/javaidnabi31/Multi-Label-Text-classification-Using-BERT/blob/master/multi-label-classification-bert.ipynb


import os
import collections
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from tools import *

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

# tf_bert_module = hub.Module("C:\\Users\\sofiene.jenzri\\Documents\\OneDrive - UiPath\\Documents\\DataScience\\project\\sources\\UniversalEncoderGoogle\\large_3")
# tf_bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
# tf_bert_module = hub.Module("C:\\Users\\sofiene.jenzri\\Documents\\OneDrive - UiPath\\Documents\\DataScience\\bert_models\\uncased_L-12_H-768_A-12")
# print (tf_bert_module.get_signature_names())
# get_signature_names()

##use downloaded model, change path accordingly

bBuildWithCuda =tf.test.is_built_with_cuda()
bGPU = tf.test.is_gpu_available(    cuda_only=True,    min_cuda_compute_capability=None)
if bGPU:
    pretrained_model_path = "D:\\MyDocs\\machineLearning\\BERT\\uncased_L-12_H-768_A-12\\"
else:
    pretrained_model_path = "C:\\Users\\sofiene.jenzri\\Documents\\OneDrive - UiPath\\Documents\\DataScience\\LM_models\\bert_models\\uncased_L-12_H-768_A-12\\"
BERT_VOCAB= pretrained_model_path + 'vocab.txt'
BERT_INIT_CHKPNT = pretrained_model_path+'bert_model.ckpt' #.data-00000-of-00001
BERT_CONFIG = pretrained_model_path+ 'bert_config.json'

tokenization.validate_case_matches_checkpoint(True,BERT_INIT_CHKPNT)
tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=True)

# print (tokenizer.tokenize("This here's an example of using the BERT tokenizer"))


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
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id,
        self.is_real_example=is_real_example

def create_examples(df, labels_available=True):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, row) in enumerate(df.values):
        guid =i # row[0]
        text_a = str(row[0])
        if labels_available:
            label = row[1]
        else:
            label = -1
        examples.append(
            InputExample(guid=guid, text_a=text_a, label=label))
    return examples


my_cats = ['rec.autos', 'soc.religion.christian', 'rec.sport.baseball', 'sci.electronics', 'sci.med']



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
    corpus_ser = corpus_ser.apply(lambda x: x if len(x.split())<256 and len(x.split())>10 else 'refused', convert_dtype=True)
    my_index = corpus_ser != 'refused'
    corpus_ser = corpus_ser[my_index].reindex()
    cats_ser =cats_ser[my_index].reindex()

    # print (corpus_ser.head())
    # print (cats_ser.head())

    doc_lens = np.array([len(doc.split()) for doc in corpus_ser])
    print (f'mean length is {np.mean(doc_lens)}')
    print (f'max length is {np.max(doc_lens)}')
    print ('raw corpus size : {}'.format(corpus_ser.size))


    data_dic = {'text': corpus_ser, 'cat':cats_ser}
    ret_df = pd.DataFrame(data_dic)
    # print(ret_df.dtypes)
    print (ret_df.head())
    return ret_df


def convert_examples_to_features(examples,  max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        print(example.text_a)
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            pass
            # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        # labels_ids = []
        # for label in example.labels:
        #     labels_ids.append(int(label))

        label_id=example.label

        if ex_index < 0:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label: %s (id = %s)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """
    
    
def convert_single_example(ex_index, example, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"

        pass
        # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

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

    # labels_ids = []
    # for label in example.labels:
    #     labels_ids.append(int(label))

    label_id = example.label

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature

def file_based_convert_examples_to_features(
        examples, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        #if ex_index % 10000 == 0:
            #tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
        # if isinstance(feature.label_ids, list):
        #     label_ids = feature.label_ids
        # else:
        #     label_ids = feature.label_ids[0]
        features["label_ids"] = create_int_feature(feature.label_id)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, rate=0.1) #keep_prob=0.9

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        
        # probabilities = tf.nn.softmax(logits, axis=-1) ### multiclass case
        # probabilities = tf.nn.sigmoid(logits)#### multi-label case
        
        # labels = tf.cast(labels, tf.float32)
        # tf.logging.info("num_labels:{};logits:{};labels:{}".format(num_labels, logits, labels))
        # per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        # loss = tf.reduce_mean(per_example_loss)

        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        #tf.logging.info("*** Features ***")
        #for name in sorted(features.keys()):
        #    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
             is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
             is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        #SJ added for logs
        logging_hook = tf.train.LoggingTensorHook({"loss" : total_loss}, every_n_iter=10)
        

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn,
                training_hooks = [logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, probabilities, is_real_example):

                # logits_split = tf.split(probabilities, num_labels, axis=-1)
                # label_ids_split = tf.split(label_ids, num_labels, axis=-1)
                # # metrics change to auc of every class
                # eval_dict = {}
                # for j, logits in enumerate(logits_split):
                #     label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                #     current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
                #     eval_dict[str(j)] = (current_auc, update_op_auc)
                # eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
                # return eval_dict

                ## original eval metrics
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = metric_fn(per_example_loss, label_ids, probabilities, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            print("mode:", mode,"probabilities:", probabilities)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold=scaffold_fn)
        return output_spec

    return model_fn


def run_classification():
    train_df = generate_raw_data_df(subset='train')
    TRAIN_VAL_RATIO = 0.9
    LEN = train_df.shape[0]
    SIZE_TRAIN = int(TRAIN_VAL_RATIO*LEN)

    x_train = train_df[:SIZE_TRAIN]
    x_val = train_df[SIZE_TRAIN:]

    # for (i, row) in enumerate(x_train.values):
    #     print (i, row[0], row[1])
    #     if i==2:
    #         break

    train_examples = create_examples(x_train, labels_available=True)
    print (train_examples[0].text_a)
    # eval_examples = create_examples(x_val, labels_available=True)
    # We'll set sequences to be at most 128 tokens long.
    MAX_SEQ_LENGTH = 256   

    # Compute train and warmup steps from batch size
    # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 1.0
    # Warmup is a period of time where hte learning rate 
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 1000 #1000
    SAVE_SUMMARY_STEPS = 5 #500

    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_examples) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
    model_path = "C:\\Users\\sofiene.jenzri\\Documents\\OneDrive - UiPath\\Documents\\DataScience\\LM_models\\bert_models\\working\\"
    train_file = os.path.join(model_path, "train.tf_record")
    #filename = Path(train_file)
    if not os.path.exists(train_file):
        open(train_file, 'w').close()

    file_based_convert_examples_to_features(
            train_examples, MAX_SEQ_LENGTH, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", BATCH_SIZE)
    tf.logging.info("  Num steps = %d", num_train_steps)

    train_input_fn = file_based_input_fn_builder( input_file=train_file,
                                                    seq_length=MAX_SEQ_LENGTH,
                                                    is_training=True,
                                                    drop_remainder=True)
    OUTPUT_DIR = model_path+"\\output"
    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(model_dir=OUTPUT_DIR,
                                        save_summary_steps=SAVE_SUMMARY_STEPS,
                                        keep_checkpoint_max=1,
                                        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)
    model_fn = model_fn_builder(bert_config=bert_config,
                                num_labels= 1,
                                init_checkpoint=BERT_INIT_CHKPNT,
                                learning_rate=LEARNING_RATE,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                use_tpu=False,
                                use_one_hot_embeddings=False)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                        config=run_config,
                                        params={"batch_size": BATCH_SIZE})

    
    tf.logging.set_verbosity(tf.logging.INFO)

    print(f'Beginning Training!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps,)
    print("Training took time ", datetime.now() - current_time)


    #Evaluation
    test_df = generate_raw_data_df(subset='test')
    test_InputExamples = test_df.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x['text'], 
                                                                   text_b = None, 
                                                                   label = x['cat']), axis = 1)

    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, [0,1,2,3,4], MAX_SEQ_LENGTH, tokenizer)
    
    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)
    eval_dic = estimator.evaluate(input_fn=test_input_fn, steps=None)
    print (eval_dic)

if __name__ == "__main__":
    # generate_raw_data_df()
    run_classification()
    
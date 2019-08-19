#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:jiangpinglei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import bert.modeling as modeling
import bert.optimization as optimization
import bert.tokenization as tokenization
import tensorflow as tf
import json
import codecs
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", None, "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float("entity_dropout", 0.1, "The initial entity labeling drop out rate.")

flags.DEFINE_float("class_dropout", 0.1, "The initial type classify drop out rate.")

flags.DEFINE_float("entity_weight", 1.0, "The initial entity labeling loss weight.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, class_label=None, entity_label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.class_label = class_label
        self.entity_label = entity_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, reference_label_ids, class_label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.reference_label_ids = reference_label_ids
        self.class_label = class_label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        lines = []
        with codecs.open(input_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                assert len(line) == 3
                lines.append(line)
            return lines


class MsparsProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")

    def get_entity_labels(self):
        return ["b", "m", "o"]

    def get_class_labels(self):
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[0].strip())
            class_label = tokenization.convert_to_unicode(line[1].strip())
            entity_label = tokenization.convert_to_unicode(line[2].strip())
            assert len(text.split()) == len(entity_label)
            examples.append(InputExample(guid=guid, text=text, class_label=class_label, entity_label=entity_label))
        return examples


def write_tokens(tokens, mode):
    if mode == "test":
        path = os.path.join(FLAGS.output_dir, "token_split.txt")
        wf = open(path, 'a', encoding='utf8')
        lines = ""
        for i in range(1, len(tokens)):
            if tokens[i] == "[SEP]":
                break
            lines += tokens[i] + ' '
        wf.write(lines + '\n')
        wf.close()


def convert_single_example(ex_index, example, entity_label_list, class_label_list, max_seq_length, tokenizer, mode):
    class_label_map = {}
    for (i, label) in enumerate(class_label_list):
        class_label_map[label] = i
    label_map = {}
    label_map_reference = {}
    for (i, label) in enumerate(entity_label_list, 1):
        label_map[label] = i
        label_map_reference[label] = i
    label_map_reference['x'] = len(entity_label_list) + 1
    label2idpath = os.path.join(FLAGS.output_dir, 'label2id.json')
    label2idpath_reference = os.path.join(FLAGS.output_dir, 'label2id_reference.json')

    if not os.path.exists(label2idpath_reference):
        with open(label2idpath_reference, 'w', encoding='utf8') as w:
            json.dump(label_map_reference, w)

    textlist = example.text.split()
    labellist = list(example.entity_label)
    tokens = []
    labels = []
    reference_labels = []
    unknow_index = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
                reference_labels.append(label_1)
            else:
                if label_1 == 'o':
                    labels.append('o')
                else:
                    labels.append('m')

                reference_labels.append('x')
            if token[m] == "[UNK]":
                unknow_index.append(i)
    assert len(tokens) == len(labels) == len(reference_labels)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
        reference_labels = reference_labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    reference_label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map['o'])
    reference_label_ids.append(label_map_reference['o'])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
        reference_label_ids.append(label_map_reference[reference_labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map['o'])
    reference_label_ids.append(label_map_reference['o'])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    class_label = class_label_map[example.class_label]

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        reference_label_ids.append(0)
        ntokens.append("**NULL**")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(reference_label_ids) == max_seq_length


    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        reference_label_ids=reference_label_ids,
        class_label=class_label
    )

    output_tokens = []
    for i, each in enumerate(ntokens):
        if each != "[UNK]":
            output_tokens.append(each)
        else:
            index = unknow_index[0]
            output_tokens.append(textlist[index])
            unknow_index = unknow_index[1:]
    write_tokens(output_tokens, mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, entity_label_list, class_label_list, max_seq_length, tokenizer, output_file, mode=None
):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, entity_label_list, class_label_list, max_seq_length, tokenizer, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features['reference_label_ids'] = create_int_feature(feature.reference_label_ids)
        features["class_label_ids"] = create_int_feature([feature.class_label])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "reference_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "class_label_ids": tf.FixedLenFeature([], tf.int64)
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def hidden2tag(hiddenlayer, numclass):
    linear = tf.keras.layers.Dense(numclass, activation=None)
    return linear(hiddenlayer)


def crf_loss(logits, labels, mask, num_labels, mask2len):
    """
    :param logits:
    :param labels:
    :param mask2len:each sample's length
    :return:
    """
    # TODO
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
            "transition",
            shape=[num_labels, num_labels],
            initializer=tf.contrib.layers.xavier_initializer()
        )

    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(logits, labels, transition_params=trans,
                                                                   sequence_lengths=mask2len)
    loss = tf.reduce_mean(-log_likelihood)

    return loss, transition


def softmax_layer(logits, labels, num_labels, mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask, dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12  # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.nn.softmax(logits, axis=-1)
    predict = tf.argmax(probabilities, axis=-1)
    return loss, predict, probabilities


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, entity_labels, entity_num_labels, class_labels, class_num_labels, use_one_hot_embeddings, entity_dropout, class_dropout, entity_weight):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    entity_output_layer = model.get_sequence_output()
    # output_layer shape is
    if is_training:
        entity_output_layer = tf.keras.layers.Dropout(rate=entity_dropout)(entity_output_layer)
    entity_logits = hidden2tag(entity_output_layer, entity_num_labels)
    # TODO test shape
    entity_logits = tf.reshape(entity_logits, [-1, FLAGS.max_seq_length, entity_num_labels])
    mask2len = tf.reduce_sum(input_mask, axis=1)
    entity_loss, trans = crf_loss(entity_logits, entity_labels, input_mask, entity_num_labels, mask2len)
    entity_predict, viterbi_score = tf.contrib.crf.crf_decode(entity_logits, trans, mask2len)
    entity_probabilities = tf.nn.softmax(entity_logits, axis=-1)

    # type classify
    class_output_layer = model.get_pooled_output()

    class_hidden_size = class_output_layer.shape[-1].value

    class_output_weights = tf.get_variable(
        "output_weights", [class_num_labels, class_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    class_output_bias = tf.get_variable(
        "output_bias", [class_num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            class_output_layer = tf.nn.dropout(class_output_layer, keep_prob=(1-class_dropout))

        class_logits = tf.matmul(class_output_layer, class_output_weights, transpose_b=True)
        class_logits = tf.nn.bias_add(class_logits, class_output_bias)
        class_probabilities = tf.nn.softmax(class_logits, axis=-1)
        class_log_probs = tf.nn.log_softmax(class_logits, axis=-1)

        one_hot_labels = tf.one_hot(class_labels, depth=class_num_labels, dtype=tf.float32)

        class_per_example_loss = -tf.reduce_sum(one_hot_labels * class_log_probs, axis=-1)
        class_loss = tf.reduce_mean(class_per_example_loss)

    loss = entity_weight * entity_loss + class_loss
    return loss, tf.argmax(class_probabilities, -1), class_probabilities, entity_predict, entity_probabilities


def model_fn_builder(bert_config, entity_num_labels, class_num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        entity_label = features["label_ids"]
        reference_label_ids = features['reference_label_ids']
        class_label = features['class_label_ids']
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, class_predict, class_probabilities, entity_predict, entity_probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, entity_label,
            entity_num_labels, class_label, class_num_labels, use_one_hot_embeddings, FLAGS.entity_dropout, FLAGS.class_dropout, FLAGS.entity_weight)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
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
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "entity_label": entity_label,
                "entity_predict": entity_predict,
                'entity_probabilities': entity_probabilities,
                'entity_reference_label_ids': reference_label_ids,
                'class_label': class_label,
                'class_predict': class_predict,
                'class_probabilities': class_probabilities
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)

        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "mspars": MsparsProcessor
    }
    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    entity_label_list = processor.get_entity_labels()
    class_label_list = processor.get_class_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        entity_num_labels=len(entity_label_list) + 1,
        class_num_labels=len(class_label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, entity_label_list, class_label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_split.txt")

        if os.path.exists(token_path):
            os.remove(token_path)
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, entity_label_list, class_label_list,
                                                 FLAGS.max_seq_length, tokenizer,
                                                 predict_file, mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)
        result = estimator.predict(input_fn=predict_input_fn)
        entity_labels, entity_predict, entity_probabilities, entity_reference, class_labels, class_predict, class_probabilities = [], [], [], [], [], [], []
        for index, each in enumerate(result):
            if index % 1000 == 0:
                tf.logging.info("Processing example: %d" % index)
            entity_labels.append(each["entity_label"])
            entity_predict.append(each["entity_predict"])
            entity_probabilities.append(each['entity_probabilities'])
            entity_reference.append(each['entity_reference_label_ids'])
            class_labels.append(each['class_label'])
            class_predict.append(each['class_predict'])
            class_probabilities.append(each['class_probabilities'])

        with codecs.open(os.path.join(FLAGS.output_dir, 'labeled_results.json'), 'w', encoding='utf8') as f:
            entity_labels = np.array(entity_labels).tolist()
            entity_predict = np.array(entity_predict).tolist()
            entity_probabilities = np.array(entity_probabilities).tolist()
            entity_reference = np.array(entity_reference).tolist()
            class_labels = np.array(class_labels).tolist()
            class_predict = np.array(class_predict).tolist()
            class_probabilities = np.array(class_probabilities).tolist()
            json.dump((entity_labels, entity_predict, entity_probabilities, entity_reference, class_labels, class_predict, class_probabilities), f, indent=2)



if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()


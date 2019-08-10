import os
import sys
import re
import numpy as np
import math
import codecs
import keras.backend.tensorflow_backend as T
import tensorflow as tf
sys.path.append(os.path.abspath('..'))
from config.basic import Config
import json
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Bidirectional, Concatenate, Dot, Activation, Lambda, Add
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from tqdm import tqdm
from keras.utils import multi_gpu_model
import random
import pickle
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from sklearn.model_selection import train_test_split
import argparse


# sequence to sequence model
class s2s():
    def __init__(self):
        self.config = Config()

    def generator(self, X, Y, shuffle):
        data_size = len(X)
        batch_num = int(data_size / self.config.batch_size)

        while True:
            if shuffle:
                X, _, Y, _ = train_test_split(X, Y, test_size=0)

            for index in range(batch_num):
                x = X[index * self.config.batch_size:(index + 1) * self.config.batch_size]
                y = Y[index * self.config.batch_size:(index + 1) * self.config.batch_size]
                encoder_source_data = np.zeros((len(x), self.config.max_length), dtype='int32')
                decoder_source_data = np.zeros((len(y), self.config.max_length), dtype='int32')
                decoder_target_data = np.zeros(
                    (len(x), self.config.max_length, self.num_tokens),
                    dtype='int32')

                for i, (xx, yy) in enumerate(zip(x, y)):
                    for j, word in enumerate(xx):
                        encoder_source_data[i, j] = self.word2index.get(word, self.word2index['UNK'])
                    # '\t' for beginning and '\n' for ending
                    for j, word in enumerate(['\t'] + yy + ['\n']):
                        decoder_source_data[i, j] = self.word2index.get(word, self.word2index['UNK'])

                        if j > 0:
                            decoder_target_data[
                                i, j - 1, self.word2index.get(word, self.word2index['UNK'])] = 1

                    decoder_target_data[i, j:, 0] = 1

                yield [encoder_source_data, decoder_source_data], decoder_target_data

    def get_data(self):
        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []
        cnt_question = Counter()
        cnt_logical = Counter()

        # get training data
        with codecs.open('new_train.json', 'r', encoding='utf8') as f:
            train = json.load(f)

            for item in tqdm(train, 'reading train'):
                logical = item['logical']
                new_logical = []

                for l in logical:
                    new_logical += re.split(r'([.:_])', l)

                self.train_X.append(item['question'])
                cnt_question.update(item['question'])
                self.train_Y.append(new_logical)
                cnt_logical.update(new_logical)
                cnt_question.update(new_logical)

        with codecs.open('new_dev.json', 'r', encoding='utf8') as f:
            test = json.load(f)

            for item in tqdm(test, 'reading dev'):
                logical = item['logical']
                new_logical = []

                for l in logical:
                    new_logical += re.split(r'([.:_])', l)

                self.test_X.append(item['question'])
                cnt_question.update(item['question'])
                self.test_Y.append(new_logical)
                cnt_logical.update(new_logical)
                cnt_question.update(new_logical)

        with codecs.open('new_test.json', 'r', encoding='utf8') as f:
            test = json.load(f)

            for item in tqdm(test, 'reading test'):
                cnt_question.update(item['question'])

        self.train_X, _, self.train_Y, _ = train_test_split(self.train_X, self.train_Y, test_size=0)
        self.word2index = dict([(w, i + 1) for i, w in enumerate([word for word, count in cnt_logical.items() if count >= 3])])
        ll = len(self.word2index)
        left = set(cnt_question.elements()) - set(self.word2index.keys())
        self.word2index.update([(w, i + 1 + ll) for i, w in enumerate([word for word in left])])
        length = len(self.word2index)
        self.word2index['\t'], self.word2index['UNK'], self.word2index['\n'] = (length + 1, length + 2, 0)
        self.index2word = dict([(index, word) for word, index in self.word2index.items()])
        self.num_tokens = len(self.word2index)
        self.encoder_words_num = len(self.word2index)
        self.decoder_words_num = ll + 1

#        with codecs.open('glove.840B.300d.txt', 'r', encoding='utf8') as f:
#            word_dict = dict()
#
#            for line in f:
#                line_list = line.strip().split(' ')
#                if len(line_list) != 301:
#                    continue
#                word_dict[line_list[0]] = np.array(list(map(float, line_list[1:])))
#
#        self.embedding_weights = np.random.rand(self.num_tokens, 300) * 2 - 1
#
#        for word, index in tqdm(self.word2index.items()):
#            if word in word_dict:
#                self.embedding_weights[index] = word_dict[word]

        with codecs.open('word2index_new.pkl', 'rb') as f:
            # pickle.dump((self.word2index, self.embedding_weights), f)
            self.word2index, self.embedding_weights = pickle.load(f)

    def transform2vector(self, sentence):
        source_data = np.zeros(self.config.max_length, dtype='int32')

        for i, word in enumerate(sentence):
            source_data[i] = self.word2index.get(word, self.word2index['UNK'])

        return source_data

    def transform2vector_one_hot(self, sentence):
        source_data = np.zeros((self.config.max_length, self.num_tokens), dtype='int32')

        for i, word in enumerate(sentence[1:]):
            source_data[i, self.word2index.get(word, self.word2index['UNK'])] = 1

        source_data[i:, 0] = 1

        return source_data

    def get_training_model(self):
        # encoder
        self.encoder_inputs = Input(shape=(None,), dtype='int32')
        self.embeddings = Embedding(self.num_tokens, 300, mask_zero=True, weights=[self.embedding_weights], trainable=False)
        self.encoder_lstm = Bidirectional(LSTM(self.config.hidden_dim, return_state=True, return_sequences=True))
        x = self.embeddings(self.encoder_inputs)
        self.encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder_lstm(x)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        self.encoder_states = [state_h, state_c]

        # decoder
        def get_zero_pad(x):
            x = tf.pad(x, [[0, 0], [0, 0], [0, self.encoder_words_num - self.decoder_words_num]])

            return x

        self.decoder_inputs = Input(shape=(None,), dtype='int32')
        self.decoder_lstm = LSTM(2 * self.config.hidden_dim, return_sequences=True, return_state=True)
        self.tanh_dense = Dense(2 * self.config.hidden_dim, activation='tanh')
        self.decoder_classifier = Dense(self.decoder_words_num, activation='softmax')
        self.pgen_dense_h = Dense(1, use_bias=False)
        self.pgen_dense_s = Dense(1, use_bias=False)
        self.pgen_dense_x = Dense(1)
        self.np = Lambda(lambda x: T.ones_like(x) - x)
        self.expand = Lambda(get_zero_pad, output_shape=lambda s: tuple(list(s)[:-1] + [self.encoder_words_num]))
        self.one_hot = Lambda(lambda x: T.one_hot(x, self.encoder_words_num),
                              output_shape=lambda input_shape: tuple(list(input_shape) + [self.encoder_words_num]))
        self.encoder_onehot = self.one_hot(self.encoder_inputs)
        embedding = self.embeddings(self.decoder_inputs)
        x, _, _ = self.decoder_lstm(embedding, initial_state=self.encoder_states)
        # attention
        y = Dot([-1, -1])([x, self.encoder_outputs])
        y = Activation('softmax')(y)
        ct = Dot([-1, -2])([y, self.encoder_outputs])
        self.pgen = Add()([self.pgen_dense_h(ct), self.pgen_dense_s(x), self.pgen_dense_x(embedding)])
        self.pgen = Activation('sigmoid')(self.pgen)
        self.npgen = self.np(self.pgen)
        self.pointer_results = Dot(axes=[-1, -2])([y, self.encoder_onehot])
        concatenated = Concatenate()([x, ct])
        decoder_outputs = self.tanh_dense(concatenated)
        decoder_outputs = self.decoder_classifier(decoder_outputs)
        decoder_outputs = self.expand(decoder_outputs)
        final_results = Lambda(lambda x: decoder_outputs * self.pgen + self.pointer_results * self.npgen)(
            decoder_outputs)
        ones_mask = T.ones_like(final_results) / float(self.encoder_words_num)
        delta_weight = 1e-4
        final_results = Lambda(lambda x: final_results * (1 - delta_weight) + ones_mask * delta_weight)(final_results)
        self.model = Model([self.encoder_inputs, self.decoder_inputs], final_results)

    def train(self):
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.model.fit_generator(self.generator(self.train_X, self.train_Y, True), validation_data=self.generator(self.test_X, self.test_Y, True), validation_steps=int(len(self.test_X) / self.config.batch_size), steps_per_epoch=int(len(self.train_X) / self.config.batch_size), epochs=self.config.epoch_num, callbacks=[ModelCheckpoint('../../pointer_model/pointer-{epoch:02d}-{val_loss:.4f}.hdf5', verbose=1, save_best_only=False, period=1, save_weights_only=True)])

    def load(self, m):
        self.model.load_weights('../../pointer_model/pointer-%s-0.01.hdf5' % m)


parser = argparse.ArgumentParser(description='get loss')
parser.add_argument('-m', '--model', help='model')
parser.add_argument('-l', '--lossdata', help='loss data file')
parser.add_argument('-g', '--gpu', help='gpu')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
c = tf.ConfigProto()
c.gpu_options.allow_growth=True
sess = tf.Session(config=c)
T.set_session(sess)
model = s2s()
model.get_data()
model.get_training_model()
# model.train()
model.load(args.model)
results = np.zeros(1)

with codecs.open('merge_%s_test.json' % args.lossdata, 'r', encoding='utf8') as f:
    loss_data = json.load(f)

cnt = 0

for X, Y in tqdm(loss_data):
    xx = model.transform2vector(X.strip().split())
    batch_size = 10
    batch_num = math.ceil(len(Y) / batch_size)
    new_out = np.zeros((len(Y), model.config.max_length))
    mean = np.zeros(len(Y))
    lengths = []

    for i in range(batch_num):
        test_X = []
        test_Y = []
        test_Y_onehot = []

        for y in Y[i * batch_size:(i + 1) * batch_size]:
            yy = model.transform2vector(['\t'] + y.strip().split() + ['\n'])
            yy_onehot = model.transform2vector_one_hot(['\t'] + y.strip().split() + ['\n'])
            test_X.append(xx)
            test_Y.append(yy)
            test_Y_onehot.append(yy_onehot)
            lengths.append(len(y.strip().split()) + 1)

        test_X = np.array(test_X, dtype='int32')
        test_Y = np.array(test_Y, dtype='int32')
        test_Y_onehot = np.array(test_Y_onehot, dtype='float32')
        output = model.model.predict([test_X, test_Y])

        for j in range(len(test_X)):
            for k in range(model.config.max_length):
                new_out[i * batch_size + j, k] = np.dot(-np.log(output[j, k]), test_Y_onehot[j][k])

    # mean = np.mean(new_out, axis=-1)
    for i in range(len(Y)):
        mean[i] = 0

        for j in range(lengths[i]):
            mean[i] += new_out[i, j]

        mean[i] /= lengths[i]

    #if min(mean[1:]) == mean[0]:
    #    cnt += 1

    # loss = tf.keras.losses.categorical_crossentropy(tf.convert_to_tensor(test_Y_onehot), tf.convert_to_tensor(output))
    #
    # loss_array = loss.eval(session=session)
    results = np.concatenate((results, mean))

results = results[1:]
results = results.tolist()
print(cnt)
print(len(loss_data))
print('acc:%.4f' % (cnt / len(loss_data)))

with codecs.open('results_test_merge_%s_model_%s.json' % (args.lossdata, args.model), 'w', encoding='utf8') as f:
    json.dump([results], f, indent=2)

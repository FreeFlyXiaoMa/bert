# %%

# ! -*- coding:utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test_stage1.csv')

# %%

# ! -*- coding:utf-8 -*-
import re, os, json, codecs, gc
import numpy as np
import pandas as pd
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

maxlen = 256
config_path = './chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/export/home/liuyuzhong/kaggle/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):  #信息token化
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)


def seq_padding(X, padding=0):  #对齐
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=16, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []


from keras.metrics import top_k_categorical_accuracy


def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(nclass, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),
                  metrics=['accuracy', acc_top2])
    print(model.summary())
    return model


from keras.utils import to_categorical

DATA_LIST = []
for data_row in train_df.iloc[:].itertuples():  #categorical
    DATA_LIST.append((data_row.text, to_categorical(data_row.label, 2)))
DATA_LIST = np.array(DATA_LIST, dtype=object)

DATA_LIST_TEST = []
for data_row in test_df.iloc[:].itertuples():
    DATA_LIST_TEST.append((data_row.text, to_categorical(0, 2)))
DATA_LIST_TEST = np.array(DATA_LIST_TEST, dtype=object)

# %%
def run_cv(nfold, data, data_label, data_test):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=520).split(data)
    train_model_pred = np.zeros((len(data), 2))
    test_model_pred = np.zeros((len(data_test), 2))

    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_valid, = data[train_fold, :], data[test_fold, :]

        model = build_bert(2)
        early_stopping = EarlyStopping(monitor='val_acc', patience=3)
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2)
        checkpoint = ModelCheckpoint('./bert_dump/' + str(i) + '.hdf5', monitor='val_acc',
                                     verbose=2, save_best_only=True, mode='max', save_weights_only=True)

        train_D = data_generator(X_train, shuffle=True)
        valid_D = data_generator(X_valid, shuffle=True)
        test_D = data_generator(data_test, shuffle=False)

        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=5,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint],
        )

        # model.load_weights('./bert_dump/' + str(i) + '.hdf5')

        # return model
        train_model_pred[test_fold, :] = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)

        del model;
        gc.collect()
        K.clear_session()

        # break

    return train_model_pred, test_model_pred


# %%

train_model_pred, test_model_pred = run_cv(5, DATA_LIST, None, DATA_LIST_TEST)

# %%

test_df['label'] = np.argmax(test_model_pred, 1)

# %%

test_df[['id', 'label']].to_csv('task2_bert_fold3.csv', index=None)

# %%
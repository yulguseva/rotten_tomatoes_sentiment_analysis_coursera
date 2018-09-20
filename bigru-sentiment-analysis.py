#Model with 2 hidden layers of Bidirectional GRU. 
#This solution aims for 63.4 accuracy on test set for 5-class text classification.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import keras
print('Keras version', keras.__version__)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import (Input, Dense, Embedding, LSTM, GRU, Flatten, 
                          Bidirectional, SpatialDropout1D,  MaxPooling1D, Concatenate, 
                        Conv1D, Dropout, BatchNormalization, Activation)
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, Nadam
from keras.models import Model, Sequential
from keras import backend as K

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train = pd.read_csv('../input/train.tsv',  sep="\t")
test = pd.read_csv('../input/test.tsv',  sep="\t")


MAX_LEN = 100
EMBEDDING_DIM = 200
MAX_FEATURES = 15000
RANDOM_STATE = 123
EPOCHS = 1
BATCH_SIZE = 128


def preprocessing(train, test, max_len=MAX_LEN, max_features=MAX_FEATURES, train_size=0.8):

    #lower
    X = train['Phrase'].apply(lambda x: x.lower())
    X_test = test['Phrase'].apply(lambda x: x.lower())
    
    #tokenizing
    X = X.values.tolist()
    X_test = X_test.values.tolist()
    X_tok = X + X_test
    tokenizer = Tokenizer(num_words=max_features, filters='')
    tokenizer.fit_on_texts(X_tok)
    
    X = tokenizer.texts_to_sequences(X)
    X_test = tokenizer.texts_to_sequences(X_test)
    
    #add zero padding to the left
    X = pad_sequences(X, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    
    word_index = tokenizer.word_index
    
    y = train['Sentiment'].values
        
    Y = to_categorical(y)
    X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                          Y,
                                                          train_size=train_size,
                                                          shuffle=True,
                                                          random_state=RANDOM_STATE,
                                                          stratify=y)
    
    return X_train, X_valid, y_train, y_valid, X_test,  word_index
    
def make_model_bigru():
    model=Sequential()
    #Input/Embedding
    model.add(Embedding(MAX_FEATURES, 200, input_length=MAX_LEN))
    
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(GRU(75, return_sequences=True)))
    model.add(Dropout(0.2))
    
    model.add(Bidirectional(GRU(36, return_sequences=False)))
    model.add(Dropout(0.2))
    
    #Dense/Output
    model.add(Dense(5,activation='softmax'))
    
    return model



print('generating validation set')
X_train, X_valid, y_train, y_valid, X_test, word_index = preprocessing(train, test)

print ('making model')
model = make_model_bigru()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

print('fitting the model')
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

print('making submisson')

sub = pd.read_csv('../input/sampleSubmission.csv')
sub['Sentiment'] = model.predict_classes(X_test, batch_size=BATCH_SIZE, verbose=1)
sub.to_csv('sub_cnn.csv', index=False)

print('End of script')

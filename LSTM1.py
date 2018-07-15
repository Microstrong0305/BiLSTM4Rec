# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import gc
from functools import partial

import keras
import numpy as np
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import SGD
from pandas import read_csv
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


max_words = 5
def get_precision_score(labels, predictions):
    score = 0
    for i in range(0, len(labels)):
        if labels[i] in predictions[i]:
            score += 1
    return score / predictions.size


def get_recall_score(labels, predictions):
    score = 0
    for i in range(0, len(labels)):
        if labels[i] in predictions[i]:
            score += 1
    return score / len(labels)

# convert an array of values into a dataset matrix
def create_dataset(dataset):
    dataX, dataY = [], []
    for line in dataset.itertuples():
        lineindex = line[0]
        thislist = dataset.loc[lineindex].values
        a = thislist[:-1]
        b = thislist[-1]
        dataX.append(a)
        dataY.append(b)
    # for i in range(len(dataset) - look_back - 1):
    #     a = dataset[i:(i + look_back), 0]
    #     dataX.append(a)
    #     b=dataset[i + look_back, 0]
    #     dataY.append(b)
    return np.array(dataX), np.array(dataY)

def InitModel():
    # create the model
    model = Sequential()
    model.add(Embedding(nb_classes, 32, input_length=max_words))
    model.add(LSTM(200,dropout=0.2, recurrent_dropout=0.2))
    # model.add(Convolution1D(filters=128, kernel_size=5, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=5))
    # model.add(Convolution1D(filters=128, kernel_size=5, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=5))
    # model.add(Convolution1D(filters=128, kernel_size=5, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=35))
    # model.add(Flatten())
    # model.add(Dense(360, kernel_initializer='normal', activation='sigmoid'))
    # model.add(Dropout(0.17))
    # model.add(Dense(150, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(60, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.32))
    # model.add(Dense(60, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.22))
    model.add(Dense(nb_classes, activation='softmax'))
    epochs = 100
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    # top20_acc = partial(keras.metrics.sparse_top_k_categorical_accuracy, k=20)
    # top20_acc.__name__ = 'top20_acc'
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

dataframe = read_csv('./datas/10thousand@6.csv', engine='python', sep='|', header=None)
nb_classes = dataframe.values.max() + 1
train, test = train_test_split(dataframe, test_size=0.2, random_state=14)

X_train, Y_train = create_dataset(train)
X_valid, Y_valid = create_dataset(test)
del dataframe, train, test;
gc.collect()

# Y_test_cat = to_categorical(Y_test, num_classes=nb_classes)
# Y_train_cat = to_categorical(Y_train, num_classes=nb_classes)

# del Y_train, Y_test;gc.collect()

my_model = InitModel()
print(my_model.summary())

'''
# plot my model
from keras.utils import plot_model

plot_model(my_model, to_file='model.png', show_shapes=True)

get_1rd_layer_output = K.function([my_model.layers[0].input, K.learning_phase()], [my_model.layers[1].output])

myoutput = K.function([my_model.layers[0].input, K.learning_phase()], [my_model.layers[7].output])
# output in test mode = 0
layer_output = myoutput([X_train, 0])[0]
# output in train mode = 1
layer_output2 = myoutput([X_train, 1])[0]
'''
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1)]
# Fit the model
my_model.fit(X_train, Y_train, batch_size=128, epochs=100, verbose=1, validation_data=(X_valid, Y_valid), callbacks=callbacks)
score = my_model.evaluate(X_valid, Y_valid, verbose=0)
print(score)

# make predictions

testPredict = my_model.predict_classes(X_valid)
testprecision = precision_score(Y_valid, testPredict, average='micro')

print('Test Score: %f Precision' % (testprecision))
y_proba = my_model.predict(X_valid, verbose=0)
y_pred = y_proba.argmax(axis=1)
k = 1
y_predTopk = np.argsort(y_proba, axis=1)[:, y_proba.shape[1] - k::]
pre = get_precision_score(Y_valid, y_predTopk)
recall = get_recall_score(Y_valid, y_predTopk)
F1 = 2 * pre * recall / (pre + recall)
print('Precision:{0}\tRecall:{1}\tF1:{2}'.format(pre, recall, F1))
score = my_model.evaluate(X_valid, Y_valid, verbose=0)
print(score)
# prediction=rcnn_model.predict_classes(X_test)
testprecision = precision_score(Y_valid, y_pred, average='micro')
print("Precision score for classification model - ", testprecision)

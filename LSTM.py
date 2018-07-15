# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import gc

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

embedding_Dimension = 100
hidden_dim_1 = 200


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
    return np.array(dataX), np.array(dataY)


dataframe = pd.read_csv('./datas/10thousand@6.csv', engine='python', sep='|', header=None)
input_length = dataframe.shape[1] - 1
nb_classes = dataframe.values.max() + 1
train, validation = train_test_split(dataframe, test_size=0.2, random_state=14)
print(len(train), len(validation))

X_train, Y_train = create_dataset(train)
X_valid, Y_valid = create_dataset(validation)
del dataframe, train, validation
gc.collect()


# my_model = InitModel(embedding_Dimension, nb_classes, input_length)
my_model = Sequential()
my_model.add(Embedding(output_dim=embedding_Dimension, input_dim=nb_classes,
                       mask_zero=False, input_length=input_length, trainable=False))
my_model.add(LSTM(hidden_dim_1, return_sequences=True))
my_model.add(Dense(nb_classes, activation='softmax'))
epochs = 100
lrate = 0.01
decay = lrate / epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
my_model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
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

# Fit the model
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1)]
my_model.fit(X_train, Y_train, batch_size=128, epochs=epochs, verbose=2, validation_data=(X_valid, Y_valid), callbacks=callbacks)
# score = my_model.evaluate(X_test, Y_test, verbose=0)
# print(score)

# make predictions

y_proba = my_model.predict(X_valid, verbose=0)
y_pred = y_proba.argmax(axis=1)
k = 20
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

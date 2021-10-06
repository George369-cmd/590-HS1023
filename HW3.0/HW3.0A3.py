#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 02:35:48 2021

@author: sunhaoxian
"""
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from keras.datasets import reuters

"""Load in and prepare data"""
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
        num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

#PARTITION DATA
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

#Multi-Class classification
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(70, activation='relu', input_shape=(10000,), 
                           kernel_regularizer=regularizers.l1(0.001)))
    model.add(layers.Dense(60, activation='relu', 
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(46, activation='softmax',
                           kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.compile(optimizer=optimizers.Adamax(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#train the model
model = build_model()
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=30,
                    batch_size=512,
                    validation_data=(x_val, y_val))

#Store the train and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#Showing plots
def plot_1():
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_2():
    plt.clf()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'ro', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
plot_1()
plot_2()


#Build a new model with an appropriate num of epochs
test_mse_score, test_mae_score = model.evaluate(x_test,one_hot_test_labels)
train_mse_score, train_mae_score = model.evaluate(x_train,one_hot_train_labels)
val_mse_score, val_mae_score = model.evaluate(x_val,y_val)
#Smaller MAE suggests better model
print('test MAE score is:',test_mae_score)
print('train MAE score is:',train_mae_score)
print('val MAE score is:',val_mae_score)



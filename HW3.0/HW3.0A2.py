#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 22:54:25 2021

@author: sunhaoxian
"""

from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

#Read in Data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


#One-Hot Encoding
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    #set specific indices of results[i] to 1
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
act = 'sigmoid'

#PARTITION DATA
fraction_train=0.9
indices = np.random.permutation(x.shape[0])
CUT=int(fraction_train*x.shape[0]); #print(CUT,x.shape,indices.shape)
training_idx, val_idx = indices[:CUT], indices[CUT:]
x_train =  x[training_idx,:]
y_train = y[training_idx]
x_val,   y_val   =  x[val_idx,:], y[val_idx]


#ANN Regression
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(24, activation='relu', input_shape=(10000,), 
                           kernel_regularizer=regularizers.l1(0.001)))
    model.add(layers.Dense(24, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

num_epochs = 20
model = build_model()
history = model.fit(x_train, y_train, 
                    epochs=num_epochs, batch_size=12, 
                    validation_data=(x_val, y_val))

#A dictionary containing data about everything that happened during training
history_dict = history.history

def plot_1():
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'ro', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def plot_2():
    plt.clf()
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, 'ro', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


plot_1()
plot_2()

#Build a new model with an appropriate num of epochs
test_mse_score, test_mae_score = model.evaluate(x_test,y_test)
train_mse_score, train_mae_score = model.evaluate(x_train,y_train)
val_mse_score, val_mae_score = model.evaluate(x_val,y_val)
#Smaller MAE suggests better model
print('test MAE score is:',test_mae_score)
print('train MAE score is:',train_mae_score)
print('val MAE score is:',val_mae_score)







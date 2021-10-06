#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 19:06:37 2021

@author: sunhaoxian
"""

import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from tensorflow.keras import optimizers
from keras.datasets import boston_housing
from tensorflow.keras import regularizers

#Load Data
(train_data, train_targets), (test_data, test_targets) =boston_housing.load_data()

#Normalize Data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

#ANN Regression
def build_model():
    model = models.Sequential()
    #A regularizer that applies a L1 regularization penalty.
    model.add(layers.Dense(60, activation='relu',
                           input_shape=(train_data.shape[1],), kernel_regularizer=regularizers.l1(0.01)))
    #A regularizer that applies a L2 regularization penalty.
    model.add(layers.Dense(60, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))
    #Create a regularizer that applies both L1 and L2 penalties.
    model.add(layers.Dense(45, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(layers.Dense(1, kernel_regularizer=regularizers.l2(0.001)))
    model.compile(optimizer=optimizers.RMSprop(lr = 0.0001), loss='mse', metrics=['mae'])
    return model

#K fold Cross Validation
k=5
num_val_samples = len(train_data) // k

num_epochs = 400
all_scores = []

for i in range(k):
    print('processing fold #', i)
    #data used to test the model
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    #data used to train the model
    partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
             axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
             axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

#validation score
print("The validation score is: ", np.mean(all_scores))

#Save the mae score of each epoch
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
             axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
             axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
    
#The validation scores at each epoch
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) 
    for i in range(num_epochs)]

#Display a smoother curve
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

#delete the data from first 50 epochs, to see more clearly
smooth_mae_history = smooth_curve(average_mae_history[50:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#Build a new model with an appropriate num of epochs
test_mse_score, test_mae_score = model.evaluate(test_data,test_targets)
train_mse_score, train_mae_score = model.evaluate(train_data,train_targets)
val_mse_score, val_mae_score = model.evaluate(val_data,val_targets)
print('test MAE score is:',test_mae_score)
print('train MAE score is:',train_mae_score)
print('val MAE score is:',val_mae_score)












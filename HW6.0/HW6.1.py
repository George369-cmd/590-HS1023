#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 02:07:38 2021

@author: sunhaoxian
"""
from keras import layers
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.datasets import fashion_mnist
from keras import Model 
from keras.models import load_model
from mpl_toolkits.mplot3d import Axes3D

INJECT_NOISE    =   False
EPOCHS          =   30
BATCH_SIZE      =   1000

(X, Y), (testX, testY) = mnist.load_data()
(fX, fY), (fTest, fLabels) = fashion_mnist.load_data()

#NORMALIZE AND RESHAPE
X = X/np.max(X) 
X = X.reshape(60000,28*28); 
testX = testX/np.max(testX)
testX = testX.reshape(10000, 28*28); 
#NORMALIZE AND RESHAPE
fX = fX/np.max(fX) 
fX = fX.reshape(60000,28*28);

#Number of Bottle Nect
n_bottleneck=80

#ADD NOISE IF DENOISING
if(INJECT_NOISE):
    EPOCHS=2*EPOCHS
    #GENERATE NOISE
    noise_factor = 0.5
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape) 
    X=X+noise
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=testX.shape) 
    testX=testX+noise
    X = np.clip(X, 0., 1.)
    testX = np.clip(testX, 0., 1.)
    
#DEEPER
model = models.Sequential()
NH=200
model.add(layers.Dense(NH, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(NH, activation='relu'))
model.add(layers.Dense(n_bottleneck, activation='relu'))
model.add(layers.Dense(NH, activation='relu'))
model.add(layers.Dense(NH, activation='relu'))
model.add(layers.Dense(28*28,  activation='linear'))


#COMPILE AND FIT
model.compile(optimizer='rmsprop',
                loss='mean_squared_error')
model.summary()
#TRAIN
history = model.fit(X, X,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(testX, testX),
                )

#Define the threshold
threshold = 4 * model.evaluate(X, X, batch_size=X.shape[0])


#HISTORY PLOT
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.legend()
plt.savefig('HW6.1-history.png')
plt.show()

#EXTRACT MIDDLE LAYER (REDUCED REPRESENTATION)
extract = Model(model.inputs, model.layers[-2].output) # Dense(128,...)
X1 = extract.predict(X)

#2D PLOT
plt.scatter(X1[:,0], X1[:,1], c=Y, cmap='tab10')
plt.savefig('HW6.1-2DPlot.png')
plt.show()

#3D PLOT
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=X1[:,0], 
    ys=X1[:,1], 
    zs=X1[:,2], 
    c=Y, 
    cmap='tab10'
)
plt.savefig('HW6.1-3DPlot.png')
plt.show()

#PLOT ORIGINAL AND RECONSTRUCTED 
X1=model.predict(X) 

#RESHAPE
X=X.reshape(60000,28,28); #print(X[0])
X1=X1.reshape(60000,28,28); #print(X[0])

#COMPARE ORIGINAL 
f, ax = plt.subplots(4,1)
I1=11; I2=46
ax[0].imshow(X[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(X[I2])
ax[3].imshow(X1[I2])
plt.savefig('HW6.1-images.png')
plt.show()

#function used to save the model to specific file
def save_model(filename):
    model.save(filename)
    
#function used to extract model from specific file
def get_model(filename):
    model = load_model(filename,compile=True)
    return model
    
save_model('HW6.1-model.h5')
model = get_model('HW6.1-model.h5')
model.summary()
history.history

#Use this model to train the fashion mnist dataset
fx_pred = model.predict(fX)

#RESHAPE
fX=fX.reshape(60000,28,28); #print(X[0])
fx_pred=fx_pred.reshape(60000,28,28); #print(X[0])

#Mean Squared Error of mnist dataset
mnist_loss = (X1-X)**2
mnist_loss = [np.mean(i) for i in mnist_loss]
#Mean Squared Error of fashion mnist dataset
f_mnist_loss = (fx_pred-fX)**2
f_mnist_loss = [np.mean(i) for i in f_mnist_loss]
mnist_fraction = 0
f_mnist_fraction = 0

for i in range(len(mnist_loss)):
    if mnist_loss[i] > threshold:
        mnist_fraction += 1
    if f_mnist_loss[i] > threshold:
        f_mnist_fraction += 1
        
mnist_fraction = mnist_fraction / len(mnist_loss)
f_mnist_fraction = f_mnist_fraction / len(f_mnist_loss)

print("The fraction of times anomalies in mnist dataset is:", mnist_fraction)
print("The fraction of times anomalies in fashion mnist dataset is:", f_mnist_fraction)


    














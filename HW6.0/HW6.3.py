#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:25:45 2021

@author: sunhaoxian
"""
from keras import layers
import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models
from keras import Model 
from keras.models import load_model
from keras.datasets import cifar10
from keras.datasets import cifar100

INJECT_NOISE    =   False
BATCH_SIZE      =   1000
NKEEP           =   5000        #DOWNSIZE DATASET
N_channels      =   3
PIX             =   32
EPOCHS          =   100 #OVERWRITE

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(fX, fY), (fTest, fLabels) = cifar100.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
fX = fX.astype('float32') / 255.

# Remove trucks from CIFAR10
truck_idx = np.where(y_train != [9])[0].tolist()
x_train = x_train[truck_idx]
truck_idx = np.where(y_test != [9])[0].tolist()
x_test = x_test[truck_idx]

x_train=x_train[0:NKEEP]
x_test=x_test[0:NKEEP]
fX=fX[0:NKEEP]


#ADD NOISE IF DENOISING
if(INJECT_NOISE):
    EPOCHS=2*EPOCHS
    #GENERATE NOISE
    noise_factor = 0.5
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_train = x_train + noise
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    x_test = x_test + noise
    x_train = np.clip(x_train, 0., 1.)
    x_test = np.clip(x_test, 0., 1.)
    
    
input_img = keras.Input(shape=(PIX, PIX, N_channels))
#ENCODER
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

#DECODER
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)

#COMPILE
model = keras.Model(input_img, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy');
model.summary()

#TRAIN
history = model.fit(x_train, x_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(x_test, x_test),
                )

#Define the threshold
threshold = 4 * model.evaluate(x_train, x_train, batch_size=x_train.shape[0])


#HISTORY PLOT
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.legend()
plt.savefig('HW6.3-history.png')
plt.show()


#PLOT ORIGINAL AND RECONSTRUCTED 
X1=model.predict(x_train) 

#RESHAPE
x_train = x_train.reshape(NKEEP,PIX,PIX, 3); #print(X[0])
X1=X1.reshape(NKEEP,PIX,PIX, 3); #print(X[0])

#COMPARE ORIGINAL 
f, ax = plt.subplots(4,1)
I1=11; I2=46
ax[0].imshow(x_train[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(x_train[I2])
ax[3].imshow(X1[I2])
plt.savefig('HW6.3-images.png')
plt.show()

#function used to save the model to specific file
def save_model(filename):
    model.save(filename)
    
#function used to extract model from specific file
def get_model(filename):
    model = load_model(filename,compile=True)
    return model
    
save_model('HW6.3-model.h5')
model = get_model('HW6.3-model.h5')
model.summary()
history.history

#Use this model to train the fashion mnist dataset
fx_pred = model.predict(fX)

#RESHAPE
fX=fX.reshape(NKEEP,PIX,PIX, 3); #print(X[0])
fx_pred=fx_pred.reshape(NKEEP,PIX,PIX, 3); #print(X[0])

#Mean Squared Error of mnist dataset
mnist_loss = (X1-x_train)**2
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

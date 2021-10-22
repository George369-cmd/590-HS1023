#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:05:34 2021

@author: sunhaoxian
"""
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers 
from tensorflow.keras import models
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

"""Commands used to control the operation of different type of dataset"""
set = 'MNIST'
#set = 'MNIST_Fashion'
#set = 'CIFAR-10'
model = 'CNN'
#model = 'ANN'
#augment = True
augment = False

#Read in train and test data
if set == 'MNIST':
    (trainX, trainY), (testX, testY) = mnist.load_data()
elif set == 'MNIST_Fashion':
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
else:
    (trainX, trainY), (testX, testY) = cifar10.load_data()

#class to represent Parameters:
class Parameters():
    def __init__(self):
        #1st layer size
        self.l1 = 32
        #2nd layer size
        self.l2 = 64
        #3rd layer size
        self.l3 = 128
        #kernel size
        self.ks = 3
        #maxPooling size
        self.ms = 2
        #dropout rate
        self.dr = 0.2
        #learning rate
        self.lr = 0.001
        #momentum
        self.mm = 0.9
        #dense layer
        self.dl = 512
        #label numbers
        self.ln = 10
        #range for generator
        self.r = 0.2
        #angle of rotatino
        self.rot = 60
        


#QUICK INFO ON IMAGE
def get_info(image):
	print("\n------------------------")
	print("INFO")
	print("------------------------")
	print("SHAPE:",image.shape)
	print("MIN:",image.min())
	print("MAX:",image.max())
	print("TYPE:",type(image))
	print("DTYPE:",image.dtype)
#	print(DataFrame(image))

print("---------------Get Info of Train-------------")
get_info(trainX)
print("---------------Get Info of Test-------------")
get_info(trainY)

#display the first 9 images
def display():
    # plot first few images
    for i in range(9):
        #define subplot
        plt.subplot(330 + 1 + i)
        #plot raw pixel data
        plt.imshow(trainX[i])
    # show the figure
    plt.show()
    
display()

#data preprocess
if set == 'MNIST' or set == 'MNIST_Fashion':
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    
# convert from integers to floats
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0
trainY = to_categorical(trainY)
testY = to_categorical(testY)

#split the train set into train and validation set
f_train=0.8; f_val=0.2
rand_indices = np.random.permutation(trainX.shape[0])
CUT1=int(f_train * trainX.shape[0]); 
CUT2=int((f_train+f_val) * trainX.shape[0]); 
train_idx, val_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2]

X_train = trainX[train_idx]
Y_train = trainY[train_idx]
X_val = trainX[val_idx]
Y_val = trainY[val_idx]


#function used to display the accuracy and loss
def show_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    #get loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'ro', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


#method used to add layers to build a CNN/ANN model
def build_model(method):
    model = models.Sequential()
    p = Parameters()
    #if the method is CNN, build CNN by adding CNN layers into the model
    if method == 'CNN':
        #layers to deal with CIFAR
        if set == 'CIFAR-10':
            model.add(Conv2D(p.l1, (p.ks, p.ks), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
            model.add(Conv2D(p.l1, (p.ks, p.ks), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(layers.MaxPooling2D((p.ms, p.ms)))
            model.add(layers.Dropout(p.dr))
            model.add(Conv2D(p.l2, (p.ks, p.ks), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(Conv2D(p.l2, (p.ks, p.ks), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(layers.MaxPooling2D((p.ms, p.ms)))
            model.add(layers.Dropout(p.dr))
            model.add(Conv2D(p.l3, (p.ks, p.ks), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(Conv2D(p.l3, (p.ks, p.ks), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(layers.MaxPooling2D((p.ms, p.ms)))
            model.add(layers.Dropout(p.dr))
            model.add(layers.Flatten())
        #layers to deal with MINST Fashion
        elif set == 'MINST_Fashion':
            model.add(Conv2D(p.l1, (p.ks, p.ks), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
            model.add(layers.MaxPooling2D((p.ms, p.ms)))
            model.add(Conv2D(p.l2, (p.ks, p.ks), activation='relu', kernel_initializer='he_uniform'))
            model.add(layers.MaxPooling2D((p.ms, p.ms)))
            model.add(Conv2D(p.l3, (p.ks, p.ks), activation='relu', kernel_initializer='he_uniform'))
            model.add(layers.MaxPooling2D((p.ms, p.ms)))
            model.add(Conv2D(p.l3, (p.ks, p.ks), activation='relu', kernel_initializer='he_uniform'))
            model.add(layers.MaxPooling2D((p.ms, p.ms)))
            model.add(layers.Flatten())
        #layers to deal with MNIST
        else:
            model.add(layers.Conv2D(p.l1, (p.ks, p.ks), activation='relu', input_shape=(28, 28, 1)))
            model.add(layers.MaxPooling2D((p.ms, p.ms)))
            model.add(layers.Conv2D(p.l2, (p.ks, p.ks), activation='relu')) 
            model.add(layers.MaxPooling2D((p.ms, p.ms)))
            model.add(layers.Conv2D(p.l2, (p.ks, p.ks), activation='relu'))
            model.add(layers.Flatten())
    #when the method is ANN, make input shae different
    else:
        if set == 'MNIST_Fashion' or set == 'MNIST':
            model.add(layers.Dense(p.dl, activation='relu', input_shape=(28, 28, 1)))
        else:
            model.add(layers.Dense(p.dl, activation='relu', input_shape=(32, 32, 3)))
    #add fully connected layers
    model.add(layers.Dense(p.l2, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(p.lr)))
    model.add(layers.Dense(p.l3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(layers.Dense(p.ln, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    

#add data augmentation to the model, train the model, and display the accuracy and loss
def fit():
    p = Parameters()
    model = build_model('CNN')
    #model = build_model('ANN')
    history = 1
    #Flag used to check whether data augmentation is applied
    if augment == True:
        #create generator for data augmentation
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        # prepare iterator
        ite = datagen.flow(X_train, Y_train, batch_size = p.l2)
        num_steps = int(X_train.shape[0] / p.l2)
        history = model.fit(ite, steps_per_epoch=num_steps, epochs=100, validation_data=(X_val, Y_val), verbose=0)
    else:
        history = model.fit(X_train, Y_train, epochs=60, batch_size=6000, validation_data=(X_val, Y_val), verbose=0)
    return model, history


trained_model, history = fit()
show_history(history)
train_loss, train_acc = trained_model.evaluate(trainX, trainY, batch_size=6000)
print('train_acc:', train_acc)
test_loss, test_acc = trained_model.evaluate(testX, testY, batch_size=6000)
print('test_acc:', test_acc)

#function used to save the model to specific file
def save_model(filename):
    trained_model.save(filename)
    
#function used to extract model from specific file
def get_model(filename):
    model = load_model(filename,compile=True)
    return model

if set == 'CIFAR-10':
    save_model('CIFAR.h5')
    trained_model = get_model('CIFAR.h5')
elif set == 'MNIST':
    save_model('MNIST.h5')
    trained_model = get_model('MNIST.h5')
else:
    save_model('Fashion.h5')
    trained_model = get_model('Fashion.h5')
   
trained_model.summary()

"""Visualizing every channel in every intermediate activation"""
def display(model):
    img_tensor = testX
    print("----------Visualizing intermediate activations-----------")
    #extract the outputs of the top eight layers
    layer_outputs = [layer.output for layer in model.layers[:8]]
    #a model that will return these outputs
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    #Returns a list of 8 Numpy arrays: one array per layer activation
    activations = activation_model.predict(img_tensor)
    #a list containing the names of each layer
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)
    #num of images per row
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        #num of features in feature map
        n_features = layer_activation.shape[-1]
        #The size of each layer
        size = layer_activation.shape[1]
        #num of rows in each layer
        n_row = n_features // images_per_row
        #a matrix full of 0's to contain the images
        display_grid = np.zeros((size * n_row, images_per_row * size))
        #Tiles each filter into a big horizontal grid
        for col in range(n_row):
            for row in range(images_per_row):
                #extract each image and normalize
                if len(layer_activation.shape) < 4:
                    break
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                #displays the grid
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        

display(trained_model)
        


        




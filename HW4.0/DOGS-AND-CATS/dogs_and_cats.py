#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:22:08 2021

@author: sunhaoxian
"""

import os, shutil
import numpy as np

"""Data Selection"""
#set the base directory
original_dataset_dir = os.getcwd() + '/train'
base_dir = os.path.join(os.getcwd(), 'cats_and_dogs_small')
os.mkdir(base_dir)

#set train directory
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

#set validation directory
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

#set test directory
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

#create cats and dogs directory in train directory
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

#create cats and dogs directory in validation directory
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

##create cats and dogs directory in test directory
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)


#copy 1000 images of cat to the train directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

#copy 500 images of cat to the validation directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

#copy 500 images of cat to the test directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)


#copy 1000 images of dogs to the train directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

#copy 500 images of dogs to the validation directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
#copy 500 images of dogs to the test directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
    
    
"""Building a baseline network"""
from keras import layers
from keras import models
model = models.Sequential()
#add convolutional network
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))

#add maxpooling to reduce dimensions
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

#get all the values at each index into an array
model.add(layers.Flatten())
#fully connected network
model.add(layers.Dense(512, activation='relu'))
#classify in 1 node
model.add(layers.Dense(1, activation='sigmoid'))

from tensorflow.keras import optimizers
#for binary classification, use RMSprop as the optimizer
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
#display the model
model.summary()


"""Data Preprocessing"""
from keras.preprocessing.image import ImageDataGenerator

#rescale all images by 255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#process all images in the target directory into a generator with size 150 * 150
#and binary label
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

#display the dimension of each batch
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


"""train the model"""
history = model.fit_generator(
        train_generator,
        #number of gradient descent steps
        steps_per_epoch=100,
        #num of epochs
        epochs=30,
        #validation step
        validation_data=validation_generator,
        validation_steps=50)

#save the model
model.save('cats_and_dogs_small_1.h5')



"""Display the accuracy and loss"""
import matplotlib.pyplot as plt
#get accuracy
def plot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    #get loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

#Obviously, the model is overfitting, so in order to avoid this, we are gonna
#apply data augmentation
plot(history)

"""Data Augmentation"""
#expose the model to more aspects of the data and generalize better
datagen = ImageDataGenerator(
        #a range within which to randomly rotate pictures
        rotation_range=40,
        #ranges within which to randomly translate pictures vertically or horizontally
        width_shift_range=0.2,
        height_shift_range=0.2,
        #randomly applying shearing transformations
        shear_range=0.2,
        #randomly zooming inside pictures
        zoom_range=0.2,
        #randomly flipping half the images horizontally—relevant 
        #when there are no assumptions of horizontal asymmetry
        horizontal_flip=True,
        #the strategy used for filling in newly created pixels
        fill_mode='nearest')

#Example:
#apply augmentation to one image
from keras.preprocessing import image
fnames = [os.path.join(train_cats_dir, fname) for
          fname in os.listdir(train_cats_dir)]
#choose one image to augment
img_path = fnames[3]
#read in image and resize it
img = image.load_img(img_path, target_size=(150, 150))
#convert the image to a numpy array
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
#display 4 augmented images
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()



"""Define a new model same as before"""
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


"""Training the convnet using data-augmentation generators"""
train_datagen = ImageDataGenerator(
        rescale=1./255,
        #range within which to randomly rotate pictures
        rotation_range=40,
        #ranges within which to randomly translate pictures vertically or horizontally
        width_shift_range=0.2,
        height_shift_range=0.2,
        #randomly applying shearing transformations
        shear_range=0.2,
        #randomly zooming inside pictures
        zoom_range=0.2,
        #randomly flipping half the images horizontally
        horizontal_flip=True,
        #the strategy used for filling in newly created pixels
        fill_mode='nearest')

#The validation set should not be augmented
test_datagen = ImageDataGenerator(rescale=1./255)

#a new train generator generated by augmentation
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

#a new validation generator generated by augmentation
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

#train/test the model with new generator
history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50)

#save the model after augmentation
model.save('cats_and_dogs_small_2.h5')

#Obviously, after data augmentation, both the accuracy and the loss have been
#largely improved
plot(history)

"""Feature Extraction"""
#Instantiating the VGG16 convolutional base
from tensorflow.keras.applications import VGG16
conv_base = VGG16(weights='imagenet',       # the weight checkpoint from which to initialize the model
                  include_top=False,        # not include the densely connected classifier on
                                            # top of the network
                  input_shape=(150, 150, 3))    #reshape

#take a look at the pretrained model
conv_base.summary()

#Extract features using the pretrained model
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

#functino to extract features from directory with sample_count
def extract_features(directory, sample_count):
    #shape of features
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    #number of labels
    labels = np.zeros(shape=(sample_count))
    #generator to iterate images from directory and reshape
    generator = datagen.flow_from_directory(
            directory,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary')
    i=0
    #iterate through the generator and extract features using predict
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        #where to stop the loop
        if i * batch_size >= sample_count:
            break
    return features, labels
    
#contain the features and labels
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

#reshape the features to feed the fully connected network
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))


"""Train the pretrained model with new data"""
from keras import models
from keras import layers
from keras import optimizers
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))
#show the accuracy and loss again
plot(history)


"""Visualizing intermediate activations"""
from keras.models import load_model
#load model from file
model = load_model('cats_and_dogs_small_2.h5')
model.summary()

img_path = os.path.join(test_cats_dir, 'cat.1700.jpg')

from keras.preprocessing import image
import numpy as np
#prepocess the image into the 4D tensor
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)

#display the original graph
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()

from keras import models
#extract the outputs of the top eight layers
layer_outputs = [layer.output for layer in model.layers[:8]]
#a model that will return these outputs
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
#Returns a list of 8 Numpy arrays: one array per layer activation
activations = activation_model.predict(img_tensor)
#get the 1st layer list
first_layer_activation = activations[0]
print(first_layer_activation.shape)
#visualization of the 4th and 7th channel
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')

"""Visualizing every channel in every intermediate activation"""
def display(model):
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
        

display(model)







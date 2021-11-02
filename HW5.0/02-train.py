#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:18:45 2021

@author: sunhaoxian
"""
#Read in Data
import os
from keras.models import Sequential 
from keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import regularizers


#Set loading directory
novel_dir = os.path.join(os.getcwd(), "novels")
train_dir = os.path.join(novel_dir, 'train')
test_dir = os.path.join(novel_dir, 'test')

#function to read in train and test text data and labels
def readIn(direct):
    labels = []
    texts = []
    for label_type in ['After_the_Manner_of_Men', 'Honor_of_Thieves', 'The_Story_of_Andre_Cornelis']:
        dir_name = os.path.join(direct, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'After_the_Manner_of_Men':
                    labels.append(0)
                elif label_type == 'Honor_of_Thieves':
                    labels.append(1)
                else:
                    labels.append(2)
    return texts, labels

train_texts, train_labels = readIn(train_dir)
test_texts, test_labels = readIn(test_dir)
                
#Vectorize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
#Cuts off texts after 500 words
maxlen = 500
#Considers only the top 10,000 words in the dataset
max_words = 10000
max_features = 10000    #DEFINES SIZE OF VOCBULARY TO USE
embed_dim    = 16        #DIMENSION OF EMBEDING SPACE (SIZE OF VECTOR FOR EACH WORD)
epochs       = 30
lr           = 0.0001    #LEARNING RATE
batch_size   = 5000
verbose = 1
#Get Test Set
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(test_texts)
sequences = tokenizer.texts_to_sequences(test_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens of test case' % len(word_index))
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(test_labels)
print('Shape of test data tensor:', x_test.shape)
print('Shape of test label tensor:', y_test.shape)

#Get Train Data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens of train case' % len(word_index))
train_data = pad_sequences(sequences, maxlen=maxlen)
train_labels = np.asarray(train_labels)
print('Shape of train data tensor:', train_data.shape)
print('Shape of train label tensor:', train_labels.shape)
indices = np.arange(train_data.shape[0])

#Split the data into train and test set
np.random.shuffle(indices)
data = train_data[indices]
labels = train_labels[indices]
training_samples = (int)(0.8 * len(data))
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: len(data)]
y_val = labels[training_samples: len(data)]


#---------------------------
#plotting function
#---------------------------
def report(history,title='',I_PLOT=True):
    print(title+": TEST METRIC (loss,accuracy):",model.evaluate(x_test,y_test,batch_size=50000,verbose=verbose))
    if(I_PLOT):
        #PLOT HISTORY
        epochs = range(1, len(history.history['loss']) + 1)
        plt.figure()
        plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
        plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')

        plt.plot(epochs, history.history['acc'], 'ro', label='Training acc')
        plt.plot(epochs, history.history['val_acc'], 'r', label='Validation acc')

        plt.title(title)
        plt.legend()
        # plt.show()

        plt.savefig('HISTORY-'+title+'.png')   # save the figure to file
        plt.show()

print("---------------------------")
print("SimpleRNN")  
print("---------------------------")
model = Sequential() 
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.SimpleRNN(32)) 
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['acc']) 
model.summary()
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
report(history,title="SimpleRNN")
model.save("SimpleRNN.h5")





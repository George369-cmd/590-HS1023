#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 17:44:07 2021

@author: sunhaoxian
"""
import os
from keras.models import load_model

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
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(test_labels)


#Get Train Data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)
word_index = tokenizer.word_index
train_data = pad_sequences(sequences, maxlen=maxlen)
train_labels = np.asarray(train_labels)
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



#Get matrics
trained_model = load_model("SimpleRNN.h5",compile=True)
train_loss, train_acc = trained_model.evaluate(x_train, y_train, batch_size=6000)
print('train accuracy:', train_acc)
print('train loss:', train_loss)
val_loss, val_acc = trained_model.evaluate(x_val, y_val, batch_size=6000)
print('validation accuracy:', val_acc)
print('validation loss:', val_loss)
test_loss, test_acc = trained_model.evaluate(x_test, y_test, batch_size=6000)
print('test accuracy:', test_acc)
print('test loss:', test_loss)
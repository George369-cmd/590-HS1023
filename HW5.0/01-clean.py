#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 19:52:54 2021

@author: sunhaoxian
"""
import nltk
import string 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


#read in honor of thieves 
honor = open("Honor of Thieves.txt").read()
#split according to paragraphs
honor = honor.split('\n\n')
after = open("After the Manner of Men.txt").read()
after = after.split('\n\n')
story = open("The Story of Andre Cornelis.txt").read()
story = story.split('\n\n')

#make directories for different sets
parent_dir = os.getcwd()
target_dir = "novels"
path = os.path.join(parent_dir, target_dir)
os.mkdir(path)
train_dir = "train"
test_dir = "test"
train_path = os.path.join(path, train_dir)
test_path = os.path.join(path, test_dir)
os.mkdir(train_path)
os.mkdir(test_path)
honor_path = "Honor_of_Thieves"
after_path = "After_the_Manner_of_Men"
story_path = "The_Story_of_Andre_Cornelis"
lemmatizer = WordNetLemmatizer()


#function to clean, split and restore the paragraphes
def clean_n_split(listOfWords, path):
    #make new directory to contain each novel in train or test set
    train_new_path = os.path.join(train_path, path)
    test_new_path = os.path.join(test_path, path)
    os.mkdir(train_new_path)
    os.mkdir(test_new_path)
    os.chdir(train_new_path)
    k = 1
    for i in range(len(listOfWords)):
        tmp_str = listOfWords[i]
        #get rid of new lines
        tmp_str = tmp_str.replace("\n", " ")
        #all to lower case
        tmp_str = tmp_str.lower()
        #lemmatize
        tmp_str = lemmatizer.lemmatize(tmp_str)
        file_name = str(i+1) + ".txt"
        #half train and half test
        if i >= (int)(len(listOfWords)*0.8):
            os.chdir(test_new_path)
            file_name = str(k) + ".txt"
            k += 1
        text_file = open(file_name, "w")
        text_file.write(tmp_str)
        text_file.close()
        
clean_n_split(honor, honor_path)
clean_n_split(after, after_path)
clean_n_split(story, story_path)

        
        
    

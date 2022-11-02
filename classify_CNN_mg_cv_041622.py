#!/usr/bin/env python

################################################
##this script is to train the data and test data
import sys
from seq_reader_kmer import load_data # read kmer feature matrices 
from sklearn.model_selection import StratifiedKFold     # cross validation
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import KFold

##import classification_report helps us to detect the accuracy for each specific class
from sklearn.metrics import classification_report

def train_model (train_seq_kmer, train_lbl_kmer):

    #######################################
    # 1. Load data into train and test sets
    train_seq, train_lbl = load_data(train_seq_kmer, train_lbl_kmer) # sequences, labels
    seq_shape = train_seq.shape[1]
    ##########################
    # 2. Preprocess input data
    print("Preprocessing input data")
    seq_train = train_seq.values.reshape(train_seq.shape[0], 1, seq_shape, 1)  ##shape[0] indicates sample number
    lbl_train_one_hot = to_categorical(train_lbl, int(2))
   
    print("Done preprocessing input data")
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
   
    ###########################
    # 3. Define model architecture
    cvscores = []
    for train, test in kfold.split(seq_train, lbl_train_one_hot):
        #create the model 
        model = Sequential()

        model.add(Conv2D(100, (1, 3), activation='relu', input_shape=(1, seq_shape, 1)))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Conv2D(150, (1, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Conv2D(225, (1, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(int(2), activation='softmax'))

        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        #Fit the model 
        model.fit(seq_train[train], lbl_train_one_hot[train], epochs=10, batch_size=100, verbose=0)

        #evaluate the model with test sets
        scores = model.evaluate(seq_train[test], lbl_train_one_hot[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

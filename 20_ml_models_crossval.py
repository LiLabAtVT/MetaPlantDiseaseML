#!/usr/bin/env python

##This script is to train and compare seven machine learning models using 
##a 10-fold cross validation with a k-mer frequency matrix
##By Marcela Johnson

####################################################################################
## input files - 1)csv file containing k-mer frequency matrix
##               2)csv file containing the labels in 0,1 form  
## output file - csv file containing the accuracies from the cross-validation
## command: python 20_ml_models_crossval.py kmer_mat.csv label.csv accuracies.csv  
####################################################################################


import sys
import scipy
import numpy as np
import matplotlib
import pandas as pd 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA #unsupervised
from sklearn.naive_bayes import GaussianNB #generative and supervised
from sklearn.neighbors import KNeighborsClassifier #discriminative and supervised
from sklearn.neural_network import MLPClassifier #discriminative and supervised
from sklearn.ensemble import RandomForestClassifier #discriminative and supervised
from sklearn.svm import SVC #discriminative and supervised
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from time import time



##Seq kmer feat input file
input_file = sys.argv[1]
##Label kmer feat input file
label_file = sys.argv[2]
##output file
out_file = sys.argv[3]

##Load seq kmer feat matrix in chunks for memory efficiency
%time chunk = pd.read_csv(input_file, header=None, chunksize=1000000)

##concat chunks of input file
%time dataset = pd.concat(chunk)

##change col name to kmer num
name_column = []
for i in range(0, len(dataset.columns)):
    name_column.append("kmer" + str(i))
dataset.columns = name_column

#read label file and change col name label 
label_df = pd.read_csv(label_file, header = None)
label = label_df.rename({0: 'label'}, axis=1)

X_train = dataset
Y_train = label
print ("shape of dataset:", X_train.shape, Y_train.shape)


##function to evaluate different ML models and returns scores from cross-val
def eval_model(model_name, seq_train, label_train, cross_val):
    scores = cross_val_score(model_name, seq_train, label_train, scoring='accuracy', cv=cross_val)
    return scores

RF_scores = eval_model(RandomForestClassifier(), X_train, Y_train.values.ravel(), 10) 
KNN_scores = eval_model(KNeighborsClassifier(), X_train, Y_train.values.ravel(), 10) 
SVM_scores = eval_model(SVC(kernel='linear'), X_train, Y_train.values.ravel(), 10) 
logreg_scores = eval_model(LogisticRegression(), X_train, Y_train.values.ravel(), 10) 
NB_scores = eval_model(GaussianNB(), X_train, Y_train.values.ravel(), 10) 
MLPNN_scores = eval_model(MLPClassifier(), X_train, Y_train.values.ravel(), 10) 
XGBoost_scores = eval_model(XGBClassifier(nthread=8), X_train, Y_train.values.ravel(), 10)

df_scores = pd.DataFrame(list(zip(RF_scores, KNN_scores, SVM_scores, logreg_scores, NB_scores, MLPNN_scores, XGBoost_scores)), 
                         columns =['RF', 'KNN', 'SVM', 'logReg', 'NB', 'MLPC', 'XGBoost'])

#print(df_scores)
df_scores.to_csv(out_file, index=False)

#!/usr/bin/env python

##This script is to train a CNN using a 10-fold cross validation with a k-mer 
##frequency matrix
##By Marcela Johnson

########################################################################################
## input files - 1)csv file containing k-mer frequency matrix
##               2)csv file containing the labels in 0,1 form  
## output file - output to the terminal
## command: python 21_pipeline_CNN_cv_041722.py kmer_mat.csv label.csv > accuracies.csv  
########################################################################################

import re
import glob
import os
import sys
import subprocess
import random
from classify_CNN_mg_cv_041622 import train_model


train_seq_kmer = sys.argv[1]
train_lbl_kmer = sys.argv[2]

train_model (train_seq_kmer, train_lbl_kmer)


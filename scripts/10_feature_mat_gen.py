#!/usr/bin/env python

##This script is to generate a k-mer frequency matrix from
##DNA sequences stored in fasta file format
##Adapted from DeepTE (https://github.com/LiLabAtVT/DeepTE) By Marcela Johnson

####################################################################################
## input file - fasta file in two blocks: >header and sequences with no new line  
## output file - two csv files, 1) kmer frequencies and 2) the labels encoded by 
##               0 (healthy) or 1(infected)
## command: python 10_feature_mat_gen.py fasta_file_sequences prefix_output_name  
####################################################################################


import sys
from numpy import savetxt
from seq_reader_kmer import load_data        # parsing data file
from one_hot_rep_kmer import generate_mats, conv_labels   # converting to correct format

#######################################
# 1. Obtain name of input file and output prefix 
# from command line
input_data_nm = 'All'
input_dataset = sys.argv[1]
output_name = sys.argv[2]

#######################################
# 2. Load data and obtain matrices
X, y = load_data(input_dataset) # sequences, labels
seq_mat = generate_mats(X)     # convert to array of representation matrices
label_mat = conv_labels(y, input_data_nm) #convert labels to 0,1

#concat prefix with rest of output file name
seq_out = output_name + '_kmer_feat.csv'
labl_out = output_name + '_label_feat.csv'

#write output to files
savetxt(seq_out, seq_mat, delimiter=',', fmt ='%.0f')
savetxt(labl_out, label_mat, delimiter=',', fmt ='%.0f')


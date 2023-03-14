# MetaPlantDiseaseML

Reference-free Plant disease detection using Machine Learning and long-read Metagenomic Sequencing

# Introduction

Surveillance for early disease detection is crucial to reduce the threat of plant diseases to food security. Metagenomic sequencing and taxonomic classification have recently been used to detect and identify plant pathogens. However, for an emerging pathogen, its genome may not be similar enough to any public genome to permit reference-based tools to identify infected samples. Also, in the case of point-of care diagnosis in the field, databases access may be limited. Therefore, here we explore reference-free detection of plant pathogens using metagenomic sequencing and machine learning (ML). We used long-read metagenomes from healthy and infected plants as our model system and constructed k-mer frequency tables to test eight different ML models. The accuracies in classifying individual reads as coming from a healthy or infected metagenome were compared. Of all models, random forest (RF) had the best combination of short run-time and high accuracy (over 0.90) using tomato metagenomes. We further evaluated the RF model with a different tomato sample infected with the same pathogen or a different pathogen and a grapevine sample infected with a grapevine pathogen and achieved similar performances. ML models can thus learn features to successfully perform reference-free detection of plant diseases whereby a model trained with one pathogen-host system can also be used to detect different pathogens on different hosts. Potential and challenges of applying ML to metagenomics in plant disease detection are discussed.

# Dependencies and Requirements

This Project is developed in Python with Modules and external tooks.

Before running this pipeline, a dependency check should be performed first to make sure every dependency is correctly installed.

For information about installing the dependencies, please see below. The version numbers listed below represents the version this pipeline is developed with, and using the newest version is recommended.

### Use Conda to Install the Required Packages (Recommended)

Install conda: https://www.anaconda.com/products/individual

Python (v3.6.11)(conda install -c conda-forge python; v3.6.11)\
Scipy (v1.5.2)(conda install -c conda-forge scipy; v1.5.2)\
Numpy (v1.19.2)(conda install -c conda-forge numpy; v1.19.2)\
Pandas (v1.1.3)(conda install -c conda-forge pandas; v1.1.3)\
Xgboost (conda install -c conda-forge xgboost; v0.90)\
Biopython (conda install -c conda-forge biopython; v1.77)\
Minimap2 (conda install -c bioconda minimap2; v2.17-r941)\
Samtools (conda install -c bioconda samtools; v1.11)\
Tensorflow (conda install tensorflow-gpu=1.14.0)\
Scikit-learn (conda install -c anaconda scikit-learn; v0.23.2)\	
Seqkit (conda install -c bioconda seqkit; GitHub; v2.0.0)

### Use pip to Install the Required Packages

Python (v3.6.11)\
Scipy (v1.5.2)\
Numpy (v1.19.2)\
Pandas (v1.1.3)\
Xgboost (v0.90)\
Biopython (v1.77)\
Minimap2 (v2.17-r941)\
Samtools (v1.11)\
Tensorflow (v1.14.0)\
Scikit-learn (v0.23.2)\	
Seqkit (v2.0.0)\

## Commands to Run the Models:

#### Step 1: Feature Matrix Generation

Command: python 10_feature_mat_gen.py fasta_file_sequences prefix_output_name\
Description: This script is to generate a k-mer frequency matrix from DNA sequences stored in fasta file format, adapted from [DeepTE](https://github.com/LiLabAtVT/DeepTE). Here, the input file is a fasta file in two blocks: >header and sequences with no new line. Two output files are generated, one with k-mer frequencies and the other with labels encoded by 0 (healthy) and 1 (infected)

#### Step 2.0: Training Machine Learning Models

Command: python 20_ml_models_crossval.py kmer_mat.csv label.csv accuracies.csv\
Description: This script is to train and compare seven machine learning models (Random Forest, K-Nearest Neighbors, Support Vector Machines, Logistic Regression, Naive Bayes, Multilayer Perceptron, XGBoost using a 10-fold cross validation with a k-mer frequency matrix. The input files for this script "kmer_mat.csv", which is the csv file containing k-mer frequency matrix and "label.csv", which is the csv file containing the labels in 0,1 form are both outputs of Step 1.0

#### Step 2.1: Training Convolutional Neural Network (CNN)

Command: python 21_pipeline_CNN_cv_041722.py kmer_mat.csv label.csv > accuracies.csv\
Description: This script is to train a Convolutional Neural Network using a 10-fold cross validation with a k-mer frequency matrix. The input files for this script "kmer_mat.csv", which is the csv file containing k-mer frequency matrix and "label.csv", which is the csv file containing the labels in 0,1 form are both outputs of Step 1.0

#### Step 3: TODO

#### Step 4: TODO

#### Step 5: TODO









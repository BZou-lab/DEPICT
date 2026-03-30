# gene_features
This folder contains the codes that was used to preprocess the raw data into the data used in training, validation, testing and inference.

## preprocessing_LINCS_from_PRNet.py
This is the code for generating the perturbational dataset used in our study. If you would like to replicate our experiments from the very beginning, you could do this. Otherwise, the already preprocessed data are in the ./Data/FinalData/adataAfterClean.h5ad.

This file preprocess the LINCS data generated from PRNet into the data we will be using in DEPICT.

Please download the data needed "Lincs_L1000.h5ad", before we do preprocess from here: https://zenodo.org/records/14230870.

The original data is provided by the paper: "Predicting transcriptional responses to novel chemical perturbations using deep generative model for drug discovery", Nat Comm, 2024.


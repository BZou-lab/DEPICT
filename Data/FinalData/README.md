# FinalData
This folder contains all the data you need to train and infer via DEPICT.

Please download the 'adataAfterClean.h5ad' and 'gptEmbed_Jul9_final.csv' before you run the model, from [Zenodo](https://zenodo.org/records/19207077).

## adataAfterClean.h5ad
The perturbational dataset derived from LINCS L1000.

## gptEmbed_Jul9_final.csv
The compound representations derived from ChatGPT API, for all compounds.

## compounds_512MFP_wholeDat_fixed.csv
The 512-bit binary Morgan fingerprints for all compounds.

## compounds_target_multihot_full.csv
The multi-hot encoded target information for all compounds. 

This multi-hot encoded target drug feature is not used in our original study, because it is considered redundant as we included the target information in the gpt-derived embeddings. If you are interested in using this target feature, you could modify the training and inference code correspondingly.
# Code
This folder contains 4 folders, which are "preprocessing_code", "train_code", "inference_code" and "downstream_analysis_code".

## Important
Please keep a track on all the paths shown in every code. Please make all the (absolute/relative) paths exist and comfortable for yourself before running any codes.

## preprocessing_code
This folder contains codes that preprocess raw data into the data used in training, validation, testing and downstream analysis. 

All data after preprocessing are in the ./Data/FinalData. So you can ignore this part if you are not interested in replicating our preprocessing step.

## train_code
This folder contains codes that train the DEPICT.

## inference_code
This folder contains codes that predict the transcriptional responses after perturbations via DEPICT. Please use it after training the model or use the model provided in the checkpoints folder.

## downstream_analysis_code
This folder contains codes of using DEPICT to carry downstream analysis.

## other_models
This folder contains codes that how we used other 2 recent machine learning models and 5 simple baselines as the comparison to DEPICT.

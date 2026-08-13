# Drug Synergy Prediction
This part corresponds to the section 'Synergy prediction using condition-matched predicted profiles' in our paper.

## Important
For all the codes, please run them interactively.

## Data
The Raw data are downloaded from the 'Supplementary data' of: 
https://aacrjournals.org/mct/article/15/6/1155/92159/An-Unbiased-Oncology-Compound-Screen-to-Identify.

## CheckData.py
This code generate some basic dataset for this downstream analysis from some shared data used in this study. Please run this code first.

## SynergyCalculation.py
This code generate the synergy label (binary: synergetic or antagonistic) for drug doublets. Please run this code second.

## CreateHT29ScreeningData.py
Run this after 'CheckData.py'. This code generate the input data for predictions using DEPICT.

## prediction_HT29_drugSynergy_meanPlate_GitHub.ipynb
Run this after running 'CheckData.py' and 'CreateHT29ScreeningData.py'. This code predicts the post-perturbational gene expression using DEPICT.

## SynergyClassification.py
Run this after all the above 4 codes. This code used two classical classifiers (random forest and ridge logistic regression) to predict the synergy label for drug doublets, using the condition-matched predicted data from DEPICT.

## SynergyClassification_ObservedLINCS.py
Run this as the last step. This code used two classical classifiers (random forest and ridge logistic regression) to predict the synergy label for drug doublets, using the condition-proxy observed data from LINCS.
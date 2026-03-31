# Exploratory Analysis
This part corresponds to the section 'Exploratory analysis on predicted perturbations' in our paper.

## Important
Please run these codes interactively.

## MakeScreenData.py
Run this first. This will create the input data for predictions using DEPICT for this downstream analysis. It will require a data called 'a549.h5ad'. The 'a549.h5ad' will be created in another downstream analysis called 'DrugPrioritization'. So if you want to run this analysis, please run the 'CreateNewDataDrugScreening_MeanBaseline.py' under the path './Code/downstream_analysis_code/DrugPrioritization' first.

## prediction_A549_166drugsMoA_GitHub.ipynb
Run this second. This will get the predicted post-perturbational gene expression by using DEPICT.

## Full2DVisual_predictedLINCS_MoATimeDosageDrug.py
Run this last. This will generate all the images shown in our paper.
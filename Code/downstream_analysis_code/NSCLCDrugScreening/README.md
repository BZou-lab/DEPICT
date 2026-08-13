# Drug Prioritization
This part corresponds to the section 'Mechanism-guided virtual screening identifies candidates that reverse NSCLC transcriptional signatures' in our paper.

## Important
For all the codes, please run them interactively.

## CreateNewDataDrugScreening_MeanBaseline.py
Run this code first. This code will generate a A549 cell line subset for later being predicted by DEPICT.

## prediction_A549_Alldrug_meanPlate_GitHub.ipynb
Run this second. This code will use DEPICT to predict the gene expression after perturbation.

## ComputeSignatureLungCancer_MeanPlate.py
This code will generate the spearman correlation and connectivity score for the observed LINCS L1000 data.

## ComputeStatisticsPrediction_PlateMean.py
This code will generate the spearman correlation and connectivity score for the predicted LINCS L1000 data from DEPICT.

## CompareTwoResults_meanPlate.py
Run this code last. This code will generate the prioritization results, and check the overlapping drugs between observed LINCS and predicticed LINCS by DEPICT.

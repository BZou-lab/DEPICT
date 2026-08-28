# TransferabilityUnseenCell
This folder contains downstream analyses for examining the effects of training set proximity on DEPICT's generalization to unseen cells. For detailed description, please see the Methods in our paper.

## DEPICT_PredictedDGE_SeparatedBySplitType.ipynb
Please note that this code is also used in the Pathway Recovery Analysis. So if you have run the Pathway Recovery Analysis, you can direct the input path into that folder so you will not run this again.

Run this first, this is the prediction code that let you extract the predicted DGE and also extract some summary statistics.

## Analysis1A_OOD_Transfer_Boundary_Audited_v1_2_GrandPlots.ipynb
Must run this first after you extracted the needed DGE, the rest three analysis you can run it in whatever order you like. 

This is the analysis for testing the effects of the baseline expression from neareast training cell on the generalization to unseen cells.

## Analysis1A_NearestResponseCellSimilarity_Audited_v3.ipynb
This is the analysis for testing the effects of the DGE from neareast training cell on the generalization to unseen cells.

## Analysis1A_DrugTrainingExposureFraction_Optimized_v3.ipynb
This is the analysis for testing the effects of the drug coverage in the training set on the generalization to unseen cells.

## Analysis1A_ConditionTrainingExposureFraction_Optimized_v3.ipynb
This is the analysis for testing the effects of the drug-dose-duration coverage in the training set on the generalization to unseen cells.

## DEPICT_Analysis1A_AllAssociation_IndividualComponents_Pearson.ipynb
Run this as the last step. This is the plot code that generates the plots in Fig 2 panel b.

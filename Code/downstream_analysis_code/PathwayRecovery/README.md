# Pathway Recovery Analysis
This folder contains the codes for pathway recovery analysis, which corresponds to the analysis in fig 3 panel b, c and d.

## DEPICT_PredictedDGE_SeparatedBySplitType.ipynb
Please note that this code is also used in the TransferabilityUnseenCell. So if you have run the TransferabilityUnseenCell, you can direct the input path into that folder so you will not run this again.

Run this first, this is the prediction code that let you extract the predicted DGE and also extract some summary statistics.

## DEPICT_Analysis3_Compute.py
Run this second, it will need the output from the previous code. This code did the hard computation for this analysis.

## DEPICT_Analysis3_PublicationPlots_PooledFig4D_fold_colors_matched_Analysis1A.ipynb
Run this last, this is the plot codes with some quality checks.

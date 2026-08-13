# DGE landscape analysis code
This part contains codes that can fully replicate the results shown in the Fig. 3 panel a. Before running the analysis, please make sure the required data are downloaded: 'adataAfterClean.h5ad' and 'gptEmbed_Jul9_final.csv'.

Before running the codes, please also modify the input and output path into your preferred path.

## DEPICT_Extract_RandomSplit_PredictedPerturbedExpression.ipynb
Run this first. This code retrieve the predictions from trained DEPICT models. These retrived predictions will later be used in analysis and plotting.

Require a big RAM and a GPU if need to speed up calculation.


## DEPICT_DGEStructure_Module2C_PooledMonteCarloTestPredictions_GlobalPredictedDGE_ImprovedVisuals_FoldMixing.ipynb
Run this second. This is the analysis for showing the reproducibilitiy of DGE organization.

## DEPICT_DGEStructure_Module2C_Hybrid600dpi_EqualPanels_SeparateLegendsColorbar.ipynb
You can ignore this. This is the plot code for generating the plots shown in the Fig. 3 panel a in our paper.


## DEPICT_Module2D_QuantitativeLocalBiologicalOrganization_compute.py
Run this after finishing Module 2C analysis, but before the last code. This is the computational codes for Module 2D analysis.


## DEPICT_Module2D_QuantitativeLocalBiologicalOrganization_posthoc_FIXED.ipynb
This is the analysis for showing the enrichment of DGE organization.

# Drug Synergy Prediction
If you are not interested in how we preprocessed to get the basic features such as the Loewe Score, you can ignore the preprocessing codes but only focus on these three codes: 'run_synergy_bootstrap_depict.py', 'run_synergy_bootstrap_observed_lincs.py' and 'quantile_synergy_bootstrap_figures_compact_metrics.ipynb'.

## Data
The Raw data are downloaded from the 'Supplementary data' of: 
https://aacrjournals.org/mct/article/15/6/1155/92159/An-Unbiased-Oncology-Compound-Screen-to-Identify.
The 'diff_geneExp_pred_ht29.csv', 'HT29_22drugs_wControl.h5ad' and 'HT29_allpairs_LoeweCI_labels.csv' are preprocessed input features into this downstream analysis.

## Preprocessing codes
### CheckData.py
This code generate some basic dataset for this downstream analysis from some shared data used in this study. Please run this code first and interactively.

### SynergyCalculation.py
This code generate the synergy label (binary: synergetic or antagonistic) for drug doublets. Please run this code second. Gives 'HT29_allpairs_LoeweCI_labels.csv'.

### CreateHT29ScreeningData.py
Run this after 'CheckData.py'. This code generate the input data for predictions using DEPICT.

### prediction_HT29_drugSynergy_meanPlate_GitHub.ipynb
Run this after running 'CheckData.py' and 'CreateHT29ScreeningData.py'. This code predicts the post-perturbational gene expression using DEPICT.

### GenerateDGE.py
Run this before the last step in preprocessing. It gives 'diff_geneExp_pred_ht29.csv'.

### GenerateFeatures.py
Run this as the last step in preprocessing. It gives 'HT29_22drugs_wControl.h5ad'.

## Synergy Prediction Codes
Once you get 'diff_geneExp_pred_ht29.csv', 'HT29_22drugs_wControl.h5ad' and 'HT29_allpairs_LoeweCI_labels.csv' (these are already stored in the GitHub, so if you are not interested in the preprocessing, just ignore previous codes), you can run these prediction codes.
### synergy_bootstrap_utils.py
The helper functions used in later building up the predictive head for drug synergy prediction.

### run_synergy_bootstrap_observed_lincs.py
The synergy prediction with bootstrapped CI on the observed but condition-mismatched LINCS transcriptomics.

### run_synergy_bootstrap_depict.py
The synergy prediction with bootstrapped CI on the condition-matched DEPICT predicted transcriptomics.

### quantile_synergy_bootstrap_figures_compact_metrics.ipynb
The plot codes that can generate the plots shown in our paper.
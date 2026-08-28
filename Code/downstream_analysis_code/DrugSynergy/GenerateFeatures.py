import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy import sparse
import ast
import re

df_diffexp_pred_read = pd.read_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/diff_geneExp_pred_ht29.csv", index_col=0) # all good

gene_df = pd.read_csv("./Data/RawData/GSE92742_Broad_LINCS_gene_info.txt", sep='\t')
lm_genes = gene_df[gene_df['pr_is_lm']==1]

drug_doublet_label = pd.read_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_allpairs_LoeweCI_labels.csv")


'''
pick out the common genes' signatures
'''
# clean up names (strip; optional: make case-insensitive)
g1 = lm_genes['pr_gene_symbol'].dropna().astype(str).str.strip()

adata_ht29_mean_baseline = sc.read('./Code/downstream_analysis_code/DrugSynergyPrediction/Data/ht29_drugScreen_meanPlate.h5ad')
adata_ht29 = sc.read("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/ht29.h5ad")

'''
subset 1, pick out the experiments in the same drugs
'''
# 1) Get the unique selected plates from the baseline AnnData
selected_plates = pd.Index(adata_ht29_mean_baseline.obs['pert_iname']).dropna().unique()

# 2) Build a mask on adata_ht29 where plate is in those selected plates
mask = adata_ht29.obs['pert_iname'].isin(selected_plates)

# 3) Subset (copy to avoid a view)
adata_ht29_subset = adata_ht29[mask].copy() # 22 drugs

'''
subset 2, pick out the experiments of the longest duration
CANNOT pick out the longest duration because some drugs only have 6 hours.
USE first subset
'''
adata_ht29_subset.obs['pert_time'].value_counts() # 1457 24 hours and 273 6 hours
adata_ht29_subset.obs['pert_dose'].value_counts() # many different dosage regimen.

adata_ht29_subset2 = adata_ht29_subset[adata_ht29_subset.obs['pert_time']==24].copy()
adata_ht29_subset2.obs['pert_iname'].nunique() # 16 drugs. cannot pick out 24 hours only.

'''
now calculate the mean differential gene expression for each drug as the drug profiles
Step 1: For each drug, for each plate, calculate the mean baseline expression over all 
        DMSO experiments in that plate, then for each plate, calculate the differential gene expression
        for every experiment with that drug. Then average the DEG for each drug over different plates to
        form one single drug profile for each drug on different duration and dosage.
Step 2: then find the closest dosage and duration for each drug in the drug synergy dataset, and use those profiles 
        as the predictors(X)
'''

'''
check the MK-2206
'''
MK2206_adata = adata_ht29_subset[adata_ht29_subset.obs['pert_iname']=='MK-2206']
MK2206_adata.obs['pert_time'].value_counts()
MK2206_adata.obs['dose'].value_counts()

'''
pick out the control experiments in the 22 drugs' plates
'''
# 1) Unique plates present in your subset
plates_keep = pd.Index(adata_ht29_subset.obs['plate']).dropna().unique()

# 2) Build masks on the full AnnData
#    - control can be 1/0 or boolean; handle both
ctrl_col = adata_ht29.obs['control']
if ctrl_col.dtype == bool:
    mask_control = ctrl_col
else:
    # treat 1 (or "1") as control; this also counts True==1
    mask_control = pd.to_numeric(ctrl_col, errors='coerce').fillna(0).astype(int).eq(1)

mask_plate = adata_ht29.obs['plate'].isin(plates_keep)
mask = mask_control & mask_plate

# 3) Subset controls
adata_ht29_controls = adata_ht29[mask].copy()

# (Optional) Quick sanity checks
print(f"Controls found on selected plates: {adata_ht29_controls.n_obs} rows")
print(adata_ht29_controls.obs['plate'].value_counts().head())

# 4) Merge controls with your subset (row-wise).
#    Use join='inner' to keep only genes common to both (typically 978).
merged_adata = ad.concat(
    [adata_ht29_subset, adata_ht29_controls],
    join='inner',
    label='source',              # adds a column in .obs named 'source'
    keys=['treated_subset','control_on_subset_plates'],
    index_unique=None            # keep original obs_names as-is
)

print(merged_adata)
merged_adata.write("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_22drugs_wControl.h5ad")
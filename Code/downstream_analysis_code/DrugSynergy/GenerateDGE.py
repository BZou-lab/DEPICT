import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy import sparse
import ast
import re

gene_df = pd.read_csv("./Data/RawData/GSE92742_Broad_LINCS_gene_info.txt", sep='\t')
lm_genes = gene_df[gene_df['pr_is_lm']==1]

'''
pick out the common genes' signatures
'''
# clean up names (strip; optional: make case-insensitive)
g1 = lm_genes['pr_gene_symbol'].dropna().astype(str).str.strip()

adata_ht29 = sc.read('./Code/downstream_analysis_code/DrugSynergyPrediction/Data/ht29_drugScreen_meanPlate.h5ad')

pred_df_ht29 = pd.read_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/pred_df_HT29_drugSynergy_plateMean_infer.csv")

'''
transfering the prediction data frame
'''
# 1) gene order from AnnData (must be 978 landmark genes)
genes = adata_ht29.var_names.tolist()
n_genes = len(genes)

# 2) robust parser for the Pred_delta cell (list/ndarray or string repr)
def parse_pred_delta(v):
    # already list-like?
    if isinstance(v, (list, np.ndarray, pd.Series)):
        return np.asarray(v, dtype=float).ravel()
    if isinstance(v, str):
        s = v.strip()
        # handle "array([...])" form
        if s.startswith("array(") and "[" in s and "]" in s:
            s = s[s.find("[")+1:s.rfind("]")]
            arr = np.fromstring(re.sub(r"[,\s]+", " ", s).strip(), sep=" ")
            return arr.astype(float)
        # try literal_eval on JSON-ish "[...]" lists
        try:
            x = ast.literal_eval(s)
            return np.asarray(x, dtype=float).ravel()
        except Exception:
            # last resort: regex all numbers
            nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", s)
            return np.asarray(nums, dtype=float).ravel()
    raise TypeError(f"Unsupported Pred_delta type: {type(v)}")

# 3) parse all Pred_delta rows into a 2D array
pred_mat = np.vstack([parse_pred_delta(v) for v in pred_df_ht29["Pred_delta"]])

# sanity check length
if pred_mat.shape[1] != n_genes:
    raise ValueError(f"Each Pred_delta must have {n_genes} values; got {pred_mat.shape[1]}.")

# 4) map adata_row -> obs rows in adata_a549 to fetch metadata
adata_rows = pd.to_numeric(pred_df_ht29["adata_row"], errors="raise").astype(int).to_numpy()
obs_names = adata_ht29.obs_names[adata_rows]            # for a readable, stable index
obs_meta  = adata_ht29.obs.iloc[adata_rows].copy()

# prefer 'dose' if it exists; otherwise use 'pert_dose' and rename to 'dose'
dose_col = "dose" if "dose" in obs_meta.columns else ("pert_dose" if "pert_dose" in obs_meta.columns else None)
if dose_col is None:
    raise KeyError("Neither 'dose' nor 'pert_dose' found in adata_a549.obs.")

meta = obs_meta[["pert_iname", dose_col, "pert_time"]].rename(columns={dose_col: "dose"})
meta.index = pd.Index(obs_names, name="obs_name")

# 5) build df_diffexp (wide: 978 gene columns + metadata)
df_diffexp = pd.DataFrame(pred_mat, index=meta.index, columns=genes)
df_diffexp = pd.concat([df_diffexp, meta], axis=1)

# quick peek
print(df_diffexp.shape)      # (n_profiles, 978 + 3)
print(df_diffexp.iloc[:2, -3:])  # shows pert_iname / dose / pert_time for first two rows

# save
df_diffexp.to_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/diff_geneExp_pred_ht29.csv",index=True, header=True)

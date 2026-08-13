import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy import sparse
import ast
import re

path = "./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/E-GEOD-18842-A-AFFY-44-analytics.tsv"

df = pd.read_csv(path, sep="\t")  # adjust 3 → correct number after checking
df = df.rename(columns={
    "Gene ID": "gene_ID",
    "Gene Name": "gene_name",
    "Design Element": "design_element",
    "'non-small cell lung cancer' vs 'normal'.p-value": "adjusted_pval",          # or "log2FoldChange": "log2FC"
    "'non-small cell lung cancer' vs 'normal'.t-statistic": "t_stat",           # or "p.value"/"pvalue": "pval"
    "'non-small cell lung cancer' vs 'normal'.log2foldchange": 'log2FC'
})

gene_df = pd.read_csv("./Data/RawData/GSE92742_Broad_LINCS_gene_info.txt", sep='\t')
lm_genes = gene_df[gene_df['pr_is_lm']==1]

'''
pick out the common genes' signatures
'''
# clean up names (strip; optional: make case-insensitive)
g1 = lm_genes['pr_gene_symbol'].dropna().astype(str).str.strip()
g2 = df['gene_name'].dropna().astype(str).str.strip()

# case-insensitive (uncomment both lines if you want it):
g1 = g1.str.upper()
g2 = g2.str.upper()

common = set(g1.unique()) & set(g2.unique())

# filter rows in df whose Gene Name is in the common set
df_common = df[df['gene_name'].astype(str).str.strip().isin(common)].copy() # 878 genes in common, missing 100 landmark genes

df_common_signi = df_common[df_common['adjusted_pval']<=0.05] # 690 genes.
df_common_signi10 = df_common[df_common['adjusted_pval']<=0.10] # 729 genes.

len(df_common_signi[df_common_signi['log2FC']<0])# 285 down-regulated genes for Tumor vs Normal
len(df_common_signi[df_common_signi['log2FC']>0])# 405 up-regulated genes for Tumor vs Normal

df_common_signi_inUse = df_common_signi.copy()
df_common_signi_inUse["reverseLog2FC"] = -df_common_signi_inUse["log2FC"]

adata_a549 = sc.read('./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/a549_drugScreen_meanPlate.h5ad')

pred_df_A549 = pd.read_csv("./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/pred_df_A549_plateMean_infer.csv")

'''
transfering the prediction data frame
'''
# 1) gene order from AnnData (must be 978 landmark genes)
genes = adata_a549.var_names.tolist()
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
pred_mat = np.vstack([parse_pred_delta(v) for v in pred_df_A549["Pred_delta"]])

# sanity check length
if pred_mat.shape[1] != n_genes:
    raise ValueError(f"Each Pred_delta must have {n_genes} values; got {pred_mat.shape[1]}.")

# 4) map adata_row -> obs rows in adata_a549 to fetch metadata
adata_rows = pd.to_numeric(pred_df_A549["adata_row"], errors="raise").astype(int).to_numpy()
obs_names = adata_a549.obs_names[adata_rows]            # for a readable, stable index
obs_meta  = adata_a549.obs.iloc[adata_rows].copy()

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
df_diffexp.to_csv("./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/diff_geneExp_pred.csv",index=True, header=True)


from scipy.stats import spearmanr

# 1) Build the signature vector (index = gene_name, values = reverseLog2FC)
sig = (
    df_common_signi_inUse
    .set_index("gene_name")["reverseLog2FC"]
)

# 2) Identify drug-profile gene columns in df_diffexp, preserving column order
meta_cols = {"pert_iname", "dose", "pert_time"}
gene_cols = [c for c in df_diffexp.columns if c not in meta_cols]

# 3) Intersect genes (preserving df_diffexp column order)
shared_genes = [g for g in gene_cols if g in sig.index]

# 4) Prepare arrays
sig_vec_full = sig.loc[shared_genes].to_numpy(dtype=float)    # signature values in shared order
drug_mat = df_diffexp.loc[:, shared_genes].to_numpy(dtype=float)  # rows = experiments, cols = shared_genes

# 5) Row-wise Spearman r (no p-values), preserving df_diffexp index
spearman_values = np.full(drug_mat.shape[0], np.nan)

for i in range(drug_mat.shape[0]):
    x = drug_mat[i, :]
    # mask finite values on both sides
    mask = np.isfinite(x) & np.isfinite(sig_vec_full)
    if mask.sum() >= 2:
        res = spearmanr(x[mask], sig_vec_full[mask])
        # SciPy returns a namedtuple with 'correlation' and 'pvalue'
        r = res.correlation if hasattr(res, "correlation") else res.statistic
        spearman_values[i] = r

# 6) Result as a Series (index kept), or DataFrame if you prefer
spearman_r = pd.Series(spearman_values, index=df_diffexp.index, name="spearman_r")

# (optional) If you want a small results DataFrame with metadata attached:
df_spearman = pd.concat([spearman_r, df_diffexp[["pert_iname", "dose", "pert_time"]]], axis=1)
df_spearman = df_spearman.sort_values(by="spearman_r", ascending=False)
# Quick sanity peek
print("Shared genes used:", len(shared_genes))
print(spearman_r.head())

df_spearman.to_csv("./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/df_spearman_pred_meanPlate.csv",index=True, header=True)

'''
results make sense
'''

'''
GSEA-style Connectivity score calculation
'''
# --- config ---
P_WEIGHT = 1.0  # exponent p for |d_i|^p weighting (GSEA-style); p=1 is standard

# --- 1) Identify gene columns & metadata (order preserved) ---
meta_cols = {"pert_iname", "dose", "pert_time"}
gene_cols = [c for c in df_diffexp.columns if c not in meta_cols]
genes = gene_cols  # preserve order exactly as in df_diffexp

# --- 2) Build up/down sets from the disease-reverse signature (using gene_name) ---
sig_df = df_common_signi_inUse[["gene_name", "reverseLog2FC"]].dropna(subset=["reverseLog2FC"])
up_set   = set(sig_df.loc[sig_df["reverseLog2FC"] > 0, "gene_name"]) # 285
down_set = set(sig_df.loc[sig_df["reverseLog2FC"] < 0, "gene_name"]) # 405

# Masks aligned to gene order in df_diffexp
up_mask_full   = np.array([g in up_set for g in genes],   dtype=bool) # np.sum(up_mask_full)==285
down_mask_full = np.array([g in down_set for g in genes], dtype=bool) # np.sum(down_mask_full)==405

# --- 3) GSEA ES on a single vector, using full background (no top-k) ---
def gsea_es_full(drug_vec: np.ndarray, set_mask: np.ndarray, p: float = 1.0):
    """
    drug_vec: 1D array (length N) of drug effects (e.g., log2FC) in the order of `genes`.
    set_mask: 1D boolean array (length N) indicating membership in the set (U' or D').
    Returns: (ES in [-1, 1], N_used, S_used)
    """
    # drop NaNs row-wise but keep global order relationship
    valid = np.isfinite(drug_vec)
    dv = drug_vec[valid]
    sm = set_mask[valid]

    N = dv.size
    S = int(sm.sum())
    if N == 0 or S == 0 or S == N:
        return np.nan, int(N), int(S)

    # rank by effect descending (largest at top)
    order = np.argsort(-dv)  # descending
    dv_sorted = dv[order]
    sm_sorted = sm[order]

    # weighted hits: |d|^p normalized to sum to +1; misses sum to -1
    hit_weights = np.abs(dv_sorted[sm_sorted]) ** p
    sum_w = hit_weights.sum()
    if sum_w > 0:
        inc_hits = hit_weights / sum_w
    else:
        # fallback: uniform if all weights are zero
        inc_hits = np.full(S, 1.0 / S, dtype=float)

    dec_miss = 1.0 / (N - S)

    steps = np.full(N, -dec_miss, dtype=float)
    hit_positions = np.flatnonzero(sm_sorted)
    steps[hit_positions] = inc_hits

    rs = np.cumsum(steps)  # running sum
    # ES is the signed extremum (max deviation from zero with sign)
    max_pos = rs.max()
    min_neg = rs.min()
    ES = max_pos if abs(max_pos) >= abs(min_neg) else min_neg

    # ES is bounded in [-1, 1] by construction
    return float(ES), int(N), int(S)

# --- 4) Compute ES_up, ES_down, and combined CS_raw per row ---
n_rows = df_diffexp.shape[0]
ES_up = np.full(n_rows, np.nan, dtype=float)
ES_down = np.full(n_rows, np.nan, dtype=float)
CS_raw = np.full(n_rows, np.nan, dtype=float)
N_used = np.zeros(n_rows, dtype=int)
n_up_used = np.zeros(n_rows, dtype=int)
n_down_used = np.zeros(n_rows, dtype=int)

drug_mat = df_diffexp.loc[:, genes].to_numpy(dtype=float)  # rows=experiments, cols=genes (order preserved)

for i in range(n_rows):
    dv = drug_mat[i, :]

    es_u, N_u, S_u = gsea_es_full(dv, up_mask_full, p=P_WEIGHT)
    es_d, N_d, S_d = gsea_es_full(dv, down_mask_full, p=P_WEIGHT)

    ES_up[i] = es_u
    ES_down[i] = es_d
    # For reporting, N should match; if NaNs differ, keep the max used
    N_used[i] = max(N_u, N_d)
    n_up_used[i] = S_u
    n_down_used[i] = S_d

    if np.isfinite(es_u) and np.isfinite(es_d):
        CS_raw[i] = 0.5 * (es_u - es_d)  # in [-1, 1]
    else:
        CS_raw[i] = np.nan

# --- 5) Assemble results DataFrame (index preserved) ---
df_connectivity = pd.DataFrame({
    "ES_up": ES_up,
    "ES_down": ES_down,
    "CS_raw": CS_raw,
    "N_background": N_used,
    "n_up_used": n_up_used,
    "n_down_used": n_down_used,
    "pert_iname": df_diffexp["pert_iname"],
    "pert_dose": df_diffexp["dose"],
    "pert_time": df_diffexp["pert_time"],
}, index=df_diffexp.index)

df_connectivity = df_connectivity.sort_values("CS_raw", ascending=False)
'''
results make sense
'''

df_connectivity.to_csv("./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/df_connectivity_pred_meanPlate.csv",index=True, header=True)




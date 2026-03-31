import pandas as pd
import numpy as np
import re
import scanpy as sc
import ast

adata_a549 = sc.read('./Code/downstream_analysis_code/ExploratoryAnalysis/Data/a549_166drugMoA.h5ad')
pred_df_A549 = pd.read_csv("./Code/downstream_analysis_code/ExploratoryAnalysis/Data/pred_df_A549_166drugsMoA_infer.csv")

drug_repur = pd.read_csv("./Data/RawData/repurposing_drugs_20200324.txt", sep='\t', skiprows=9)
drug_moa = drug_repur.copy()
drug_moa = drug_moa[drug_moa['moa'].notna()]
drug_moa['moa'].nunique() # 1436 kinds of moa.
moa_table = drug_moa['moa'].value_counts()

# Start from rows with non-missing MoA
df = drug_moa[drug_moa['moa'].notna()].copy()

# --- Quick diagnostic: see what "alk" looks like in your column ---
examples_alk = df[df['moa'].str.contains(r'\balk\b|anaplastic lymphoma kinase', case=False, na=False)]
print("Found ALK-like rows:", len(examples_alk))
print(examples_alk['moa'].head(10).to_list())

# --- Define NSCLC-relevant MoA patterns (case-insensitive) ---
nsclc_patterns = {
    # ---- original 7 ----
    "EGFR inhibitor": r'(?:\begfr\b|\berbb1\b).*inhib|inhib.*(?:\begfr\b|\berbb1\b)',
    "ALK inhibitor": r'(?:\balk\b|anaplastic lymphoma kinase).*inhib|inhib.*(?:\balk\b|anaplastic lymphoma kinase)',
    "MEK inhibitor": r'(?:\bmek\b|\bmek1\b|\bmek2\b|\bmap2k1\b|\bmap2k2\b).*inhib|inhib.*(?:\bmek\b|\bmap2k[12]\b)',
    "PI3K inhibitor": r'(?:\bpi3k\b|\bpik3[a-d]?\b).*inhib|inhib.*(?:\bpi3k\b|\bpik3[a-d]?\b)',
    "Topoisomerase inhibitor": r'(?:\btopoisomerase\b|\btop[12]?\b).*inhib|inhib.*(?:\btopoisomerase\b|\btop[12]?\b)',
    "mTOR inhibitor": r'(?:\bmtor\b).*inhib|inhib.*(?:\bmtor\b)',
    "HDAC inhibitor": r'(?:\bhdac\b).*inhib|inhib.*(?:\bhdac\b)',

    # ---- added NSCLC-relevant MoAs ----
    # MET / HGFR
    "MET inhibitor": r'(?:\bmet\b|\bhgfr\b|hepatocyte growth factor receptor|c-?met).*inhib|inhib.*(?:\bmet\b|\bhgfr\b|hepatocyte growth factor receptor|c-?met)',
    # RET
    "RET inhibitor": r'(?:\bret\b).*inhib|inhib.*(?:\bret\b)',
    # RAF (incl. BRAF, RAF1/CRAF)
    "RAF inhibitor": r'(?:\braf\b|\bbraf\b|\braf1\b|\bcraf\b).*inhib|inhib.*(?:\braf\b|\bbraf\b|\braf1\b|\bcraf\b)',
    # ERK (MAPK1/3)
    "ERK inhibitor": r'(?:\berk\b|\bmapk1\b|\bmapk3\b).*inhib|inhib.*(?:\berk\b|\bmapk1\b|\bmapk3\b)',
    # KRAS
    "KRAS inhibitor": r'(?:\bk-?ras\b|\bkras\b).*inhib|inhib.*(?:\bk-?ras\b|\bkras\b)',
    # FGFR family
    "FGFR inhibitor": r'(?:\bfgfr\b|\bfgfr[1-4]\b).*inhib|inhib.*(?:\bfgfr\b|\bfgfr[1-4]\b)',
    # AXL
    "AXL inhibitor": r'(?:\baxl\b).*inhib|inhib.*(?:\baxl\b)',
    # VEGF/VEGFR axis (VEGFR1 FLT1, VEGFR2 KDR, VEGFR3 FLT4)
    "VEGFR inhibitor": r'(?:\bvegfr\b|\bflt1\b|\bkdr\b|\bflt4\b|\bvegf\b).*inhib|inhib.*(?:\bvegfr\b|\bflt1\b|\bkdr\b|\bflt4\b|\bvegf\b)',
    "Angiogenesis inhibitor": r'\bangiogen(?:esis|ic)\b.*inhib|inhib.*\bangiogen(?:esis|ic)\b',

    # Cytotoxic backbones
    "Microtubule inhibitor": r'(?:\bmicrotubule\b|\btubulin\b).*inhib|inhib.*(?:\bmicrotubule\b|\btubulin\b)|tubulin polymerization inhibitor|microtubule stabilizing agent',
    "Thymidylate synthase inhibitor": r'(?:\bts\b|\bthy midylate synthase\b|\bthymidylate synthase\b|\btyms\b).*inhib|inhib.*(?:\btyms\b|\bthymidylate synthase\b|\bts\b)',
    "Alkylating / Cross-linking agent": r'(?:\bdna\b.*(?:alkylat|cross-?link)|(?:alkylat|cross-?link).*\bdna\b)',

    # DDR / combinations
    "PARP inhibitor": r'(?:\bparp\b).*inhib|inhib.*(?:\bparp\b)',
    "WEE1 inhibitor": r'(?:\bwee1\b).*inhib|inhib.*(?:\bwee1\b)',
}


# Also match variants like "EGFR/ALK inhibitor", "EGFR-ALK dual inhibitor", etc.
# The patterns above already handle "X ... inhibit" OR "inhibit ... X" and allow slashes/hyphens via ".*"

# --- Build boolean masks for each NSCLC class ---
match_masks = {
    label: df['moa'].str.contains(pat, case=False, regex=True, na=False)
    for label, pat in nsclc_patterns.items()
}

# --- Keep rows that match ANY of the NSCLC patterns ---
any_match_mask = pd.Series(False, index=df.index)
for m in match_masks.values():
    any_match_mask |= m

df_nsclc = df[any_match_mask].copy()
print(f"NSCLC-relevant rows: {len(df_nsclc)} of {len(df)}")

# --- Assign a clean NSCLC MoA label (first matching label by the dict order) ---
def assign_nsclc_label(text: str) -> str:
    t = "" if pd.isna(text) else str(text)
    for label, pat in nsclc_patterns.items():
        if re.search(pat, t, flags=re.IGNORECASE):
            return label
    return None  # shouldn't happen if filtered by any_match_mask

df_nsclc['moa_nsclc'] = df_nsclc['moa'].apply(assign_nsclc_label)

# count frequencies
moa_counts = df_nsclc["moa_nsclc"].value_counts()

# keep only those with >=10
valid_moas = moa_counts[moa_counts >= 10].index

# filter
df_nsclc = df_nsclc[
    df_nsclc["moa_nsclc"].isin(valid_moas)
].copy()

df_nsclc["moa_nsclc"].value_counts()
df_nsclc["moa_nsclc"].nunique() # 16

moa_to_group = {
    # DNA damage & repair
    'PARP inhibitor': 'DNA damage & repair',
    'Thymidylate synthase inhibitor': 'DNA damage & repair',
    'Alkylating / Cross-linking agent': 'DNA damage & repair',

    'Topoisomerase inhibitor': 'Topoisomerase inhibitor',

    # Mitotic / cell-cycle
    'Microtubule inhibitor': 'Mitotic / cell-cycle',

    # RTK oncogenic drivers
    'EGFR inhibitor': 'RTK oncogenic drivers (EGFP/ALK/MET/FGFR)',
    'ALK inhibitor': 'RTK oncogenic drivers (EGFP/ALK/MET/FGFR)',
    'MET inhibitor': 'RTK oncogenic drivers (EGFP/ALK/MET/FGFR)',
    'FGFR inhibitor': 'RTK oncogenic drivers (EGFP/ALK/MET/FGFR)',

    # Angiogenesis / VEGF axis
    'VEGFR inhibitor': 'Angiogenesis / VEGF axis',
    'Angiogenesis inhibitor': 'Angiogenesis / VEGF axis',

    # MAPK
    'MEK inhibitor': 'MAPK (RAF→MEK)',
    'RAF inhibitor': 'MAPK (RAF→MEK)',

    # PI3K–AKT–mTOR
    'PI3K inhibitor': 'PI3K–AKT–mTOR',
    'mTOR inhibitor': 'PI3K–AKT–mTOR',

    # Epigenetic
    'HDAC inhibitor': 'HDAC inhibitor',
}

order = [
    'DNA damage & repair',
    'Topoisomerase inhibitor',
    'Mitotic / cell-cycle',
    'RTK oncogenic drivers (EGFP/ALK/MET/FGFR)',
    'Angiogenesis / VEGF axis',
    'MAPK (RAF→MEK)',
    'PI3K–AKT–mTOR',
    'HDAC inhibitor',
]

df_nsclc['moa_group'] = df_nsclc['moa_nsclc'].map(moa_to_group).fillna('Other')
df_nsclc['moa_group'] = pd.Categorical(df_nsclc['moa_group'], categories=order, ordered=True)

df_nsclc.to_csv("./Code/downstream_analysis_code/ExploratoryAnalysis/Data/DrugsOfInterestMoA.csv")




'''
process the prediction matrix into a data frame
'''
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
df_diffexp = pd.DataFrame(pred_mat, index=meta.index)
df_diffexp = pd.concat([df_diffexp, meta], axis=1)


df_diffexp = df_diffexp.copy()
df_nsclc  = df_nsclc.copy()
df_diffexp['pert_iname_norm'] = df_diffexp['pert_iname'].str.strip().str.lower()
df_nsclc['pert_iname_norm']   = df_nsclc['pert_iname'].str.strip().str.lower()

common_drugs = set(df_nsclc['pert_iname_norm'])
df_pred_nsclc = df_diffexp[df_diffexp['pert_iname_norm'].isin(common_drugs)].copy()

print(f"Rows kept: {len(df_pred_nsclc)}")
print(f"Unique drugs: {df_pred_nsclc['pert_iname_norm'].nunique()}")
# build a 1-row-per-drug label table from df_nsclc
label_cols = ['pert_iname_norm']
if 'moa_group' in df_nsclc.columns:
    label_cols.append('moa_group')
if 'moa_nsclc' in df_nsclc.columns:
    label_cols.append('moa_nsclc')
if 'moa' in df_nsclc.columns:
    label_cols.append('moa')

df_labels = (df_nsclc[label_cols]
             .dropna(subset=['pert_iname_norm'])
             .drop_duplicates('pert_iname_norm'))

# left-merge labels onto the predictions
df_pred_nsclc = df_pred_nsclc.merge(
    df_labels, on='pert_iname_norm', how='left', validate='m:1'
)

# quick sanity check
print(df_pred_nsclc[['pert_iname','moa_nsclc','moa_group']].head() if 'moa_group' in df_pred_nsclc.columns
      else df_pred_nsclc[['pert_iname','moa']].head())

df_pred_nsclc['moa_group'].nunique()
df_pred_nsclc['moa_group'].value_counts()

df_pred_nsclc.to_csv("./Code/downstream_analysis_code/ExploratoryAnalysis/Data/df_deg_nsclc_a549.csv")



# --- add this import near your other imports ---
import matplotlib
matplotlib.use("Agg")
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ... your metadata detection code stays the same ...
# --- 0) Helper to normalize a column name to lowercased string
def _norm_colname(c):
    try:
        return str(c).strip().lower()
    except Exception:
        return str(c)

# --- 1) Identify feature columns robustly ---
META_CANDIDATES = {
    'pert_iname','pert_time','dose','moa_nsclc','moa',
    'pert_iname_norm','plate','well', 'moa_group'
}
meta_cols = [c for c in df_pred_nsclc.columns if _norm_colname(c) in META_CANDIDATES]

# take all numeric columns, then drop any that are known metadata by name
num_cols  = df_pred_nsclc.select_dtypes(include=[np.number]).columns.tolist()
gene_cols = [c for c in num_cols if _norm_colname(c) not in META_CANDIDATES]
assert len(gene_cols) > 0, "No numeric feature columns detected (after excluding metadata)."

# --- 2) Labels (best available) ---
label_col = 'moa_group' if 'moa_group' in df_pred_nsclc.columns else ('moa_nsclc' if 'moa_nsclc' in df_pred_nsclc.columns else None)
labels = df_pred_nsclc[label_col].astype(str).values if label_col else None

# --- 3) Matrix + scaling (same as before) ---
X = df_pred_nsclc[gene_cols].to_numpy(dtype=float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
scaler = StandardScaler(with_mean=True, with_std=True)
Xz = scaler.fit_transform(X)

# --- 4) PCA (same idea; keep ~50 PCs) ---
n_pcs = min(50, Xz.shape[1])
Xpca = PCA(n_components=n_pcs, random_state=0).fit_transform(Xz)

# ========= NEW: Build kNN graph explicitly (cosine) and run UMAP from it =========
# Create a tiny AnnData just to leverage Scanpy’s neighbor graph + UMAP
adata = sc.AnnData(Xpca)                       # rows = drugs, columns = PCs
# Build the neighbor graph in this PCA space (cosine usually best for DEGs)
K = 30  # try 15, 30, 50 to see which gives best MoA separation
sc.pp.neighbors(adata, n_neighbors=K, metric="cosine", use_rep="X")

# UMAP on the precomputed neighbor graph; tune min_dist for tighter vs looser clusters
sc.tl.umap(adata, min_dist=0.50, spread=2.0)   # try min_dist in [0.05, 0.5]
emb = adata.obsm["X_umap"]                     # (N, 2) UMAP coordinates

# --- 5) Plot (unchanged styling, but using 'emb' computed from the graph) ---
plt.figure(figsize=(9,7))
label_col = "moa_group" if "moa_group" in df_pred_nsclc.columns else (
    "moa_nsclc" if "moa_nsclc" in df_pred_nsclc.columns else "moa"
)
labels = df_pred_nsclc[label_col].astype(str).values

codes, uniques = pd.factorize(labels)
n_cls = len(uniques)
cmap = mpl.cm.get_cmap('tab20', n_cls) if n_cls > 10 else mpl.cm.get_cmap('tab10', n_cls)
colors = cmap(codes)

scat = plt.scatter(emb[:,0], emb[:,1], c=colors, s=16, alpha=0.85, linewidths=0)

handles = [Line2D([0],[0], marker='o', linestyle='',
                  markersize=6, markerfacecolor=cmap(i), markeredgecolor='none')
           for i in range(n_cls)]
legend_title = "MoA"
plt.legend(handles, uniques, title=legend_title, bbox_to_anchor=(1.02,1),
           loc="upper left", borderaxespad=0., ncol=1, fontsize=9)

plt.title(f"Grouped by Drug MoA")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.tight_layout()

outfile = "./Code/downstream_analysis_code/ExploratoryAnalysis/Figs/PredictedLINCS_umap_nsclc_8MoAGroups_graphfirst_k30_mindist05Spread2_compare.pdf"
plt.savefig(outfile, bbox_inches="tight")
plt.close()

### label by dose
# df_pred_nsclc['dose'].nunique() # 74 unique dosages. cannot do label group because too many
meta_cols = [c for c in df_pred_nsclc.columns if _norm_colname(c) in META_CANDIDATES]

# take all numeric columns, then drop any that are known metadata by name
num_cols  = df_pred_nsclc.select_dtypes(include=[np.number]).columns.tolist()
gene_cols = [c for c in num_cols if _norm_colname(c) not in META_CANDIDATES]
assert len(gene_cols) > 0, "No numeric feature columns detected (after excluding metadata)."

# --- 2) Labels (best available) ---
label_col = 'dose'
labels = df_pred_nsclc[label_col].astype(str).values if label_col else None

# --- 3) Matrix + scaling (same as before) ---
X = df_pred_nsclc[gene_cols].to_numpy(dtype=float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
scaler = StandardScaler(with_mean=True, with_std=True)
Xz = scaler.fit_transform(X)

# --- 4) PCA (same idea; keep ~50 PCs) ---
n_pcs = min(50, Xz.shape[1])
Xpca = PCA(n_components=n_pcs, random_state=0).fit_transform(Xz)

# ========= NEW: Build kNN graph explicitly (cosine) and run UMAP from it =========
# Create a tiny AnnData just to leverage Scanpy’s neighbor graph + UMAP
adata = sc.AnnData(Xpca)                       # rows = drugs, columns = PCs
# Build the neighbor graph in this PCA space (cosine usually best for DEGs)
K = 30  # try 15, 30, 50 to see which gives best MoA separation
sc.pp.neighbors(adata, n_neighbors=K, metric="cosine", use_rep="X")

# UMAP on the precomputed neighbor graph; tune min_dist for tighter vs looser clusters
sc.tl.umap(adata, min_dist=0.50, spread=2.0)   # try min_dist in [0.05, 0.5]
emb = adata.obsm["X_umap"]                     # (N, 2) UMAP coordinates

# --- 5) Plot with continuous colormap by numeric dose ---
dose = pd.to_numeric(df_pred_nsclc['dose'], errors='coerce')

# optional: clip extremes to avoid a few outliers dominating the scale
vmin, vmax = np.nanpercentile(dose, [2, 98])

# if doses are >0 and span orders of magnitude, LogNorm is nice; otherwise use Normalize
use_log = np.isfinite(dose).all() and (dose.min() > 0) and (dose.max() / dose.min() >= 50)
norm = mpl.colors.LogNorm(vmin=max(vmin, np.finfo(float).eps), vmax=vmax) if use_log else mpl.colors.Normalize(vmin=vmin, vmax=vmax)

plt.figure(figsize=(9,7))
scat = plt.scatter(emb[:,0], emb[:,1],
                   c=dose, cmap='viridis', norm=norm,
                   s=16, alpha=0.85, linewidths=0)

# mark NaN doses (if any) in gray on top
nan_mask = dose.isna().values
if nan_mask.any():
    plt.scatter(emb[nan_mask,0], emb[nan_mask,1], c='lightgray', s=16, alpha=0.85, linewidths=0, label='dose NA')
    plt.legend(loc='upper left')

cbar = plt.colorbar(scat, shrink=0.9)
cbar.set_label('Dose')

plt.title(f"Grouped by Drug Dosage")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")

plt.tight_layout()

outfile = "./Code/downstream_analysis_code/ExploratoryAnalysis/Figs/PredictedLINCS_umap_nsclc_Dose_graphfirst_k30_mindist05Spread2_compare.pdf"
plt.savefig(outfile, bbox_inches="tight")
plt.close()

### label by duration
meta_cols = [c for c in df_pred_nsclc.columns if _norm_colname(c) in META_CANDIDATES]

# take all numeric columns, then drop any that are known metadata by name
num_cols  = df_pred_nsclc.select_dtypes(include=[np.number]).columns.tolist()
gene_cols = [c for c in num_cols if _norm_colname(c) not in META_CANDIDATES]
assert len(gene_cols) > 0, "No numeric feature columns detected (after excluding metadata)."

# --- 2) Labels (best available) ---
label_col = 'pert_time'
labels = df_pred_nsclc[label_col].astype(str).values if label_col else None

# --- 3) Matrix + scaling (same as before) ---
X = df_pred_nsclc[gene_cols].to_numpy(dtype=float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
scaler = StandardScaler(with_mean=True, with_std=True)
Xz = scaler.fit_transform(X)

# --- 4) PCA (same idea; keep ~50 PCs) ---
n_pcs = min(50, Xz.shape[1])
Xpca = PCA(n_components=n_pcs, random_state=0).fit_transform(Xz)

# ========= NEW: Build kNN graph explicitly (cosine) and run UMAP from it =========
# Create a tiny AnnData just to leverage Scanpy’s neighbor graph + UMAP
adata = sc.AnnData(Xpca)                       # rows = drugs, columns = PCs
# Build the neighbor graph in this PCA space (cosine usually best for DEGs)
K = 30  # try 15, 30, 50 to see which gives best MoA separation
sc.pp.neighbors(adata, n_neighbors=K, metric="cosine", use_rep="X")

# UMAP on the precomputed neighbor graph; tune min_dist for tighter vs looser clusters
sc.tl.umap(adata, min_dist=0.50, spread=2.0)   # try min_dist in [0.05, 0.5]
emb = adata.obsm["X_umap"]                     # (N, 2) UMAP coordinates

# --- 5) Plot (unchanged styling, but using 'emb' computed from the graph) ---
plt.figure(figsize=(9,7))
label_col = "pert_time"
labels = df_pred_nsclc[label_col].astype(str).values

codes, uniques = pd.factorize(labels)
n_cls = len(uniques)
cmap = mpl.cm.get_cmap('tab20', n_cls) if n_cls > 10 else mpl.cm.get_cmap('tab10', n_cls)
colors = cmap(codes)

scat = plt.scatter(emb[:,0], emb[:,1], c=colors, s=16, alpha=0.85, linewidths=0)

handles = [Line2D([0],[0], marker='o', linestyle='',
                  markersize=6, markerfacecolor=cmap(i), markeredgecolor='none')
           for i in range(n_cls)]
legend_title = "Perturbation Duration"
plt.legend(handles, uniques, title=legend_title, bbox_to_anchor=(1.02,1),
           loc="upper left", borderaxespad=0., ncol=1, fontsize=9)

plt.title(f"Grouped by Perturbation Duration")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.tight_layout()

outfile = "./Code/downstream_analysis_code/ExploratoryAnalysis/Figs/PredictedLINCS_umap_nsclc_Duration_graphfirst_k30_mindist05Spread2_compare.pdf"
plt.savefig(outfile, bbox_inches="tight")
plt.close()


'''
inspect outliers
'''
# 1) attach UMAP to the dataframe
df = df_pred_nsclc.copy()
df["umap1"] = emb[:, 0]
df["umap2"] = emb[:, 1]

# 2) define your “red circle” box (tweak as needed)
mask = df["umap2"].between(-5, 5) & (df["umap1"] < -10)

# (optional) restrict to the Topoisomerase group only
# mask = mask & df["moa_group"].astype(str).eq("Topoisomerase inhibitor")

# 3) inspect the hits
cols = ["pert_iname","dose","pert_time","moa_group","moa_nsclc","moa","umap1","umap2"]
hits = df.loc[mask, cols].sort_values(["umap1","umap2"])
print(f"Matched {len(hits)} samples")
print(hits.to_string(index=False))
'''
the outliars are all camptothecin.
'''

'''
now plot only the topoisomerase inhibitors for Dr. He.
'''
### group by drug names
df_topoi = df_pred_nsclc[df_pred_nsclc['moa_group']=='Topoisomerase inhibitor']
meta_cols = [c for c in df_topoi.columns if _norm_colname(c) in META_CANDIDATES]

# take all numeric columns, then drop any that are known metadata by name
num_cols  = df_topoi.select_dtypes(include=[np.number]).columns.tolist()
gene_cols = [c for c in num_cols if _norm_colname(c) not in META_CANDIDATES]
assert len(gene_cols) > 0, "No numeric feature columns detected (after excluding metadata)."

# --- 2) Labels (best available) ---
label_col = 'pert_iname'
labels = df_topoi[label_col].astype(str).values if label_col else None

# --- 3) Matrix + scaling (same as before) ---
X = df_topoi[gene_cols].to_numpy(dtype=float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
scaler = StandardScaler(with_mean=True, with_std=True)
Xz = scaler.fit_transform(X)

# --- 4) PCA (same idea; keep ~50 PCs) ---
n_pcs = min(50, Xz.shape[1])
Xpca = PCA(n_components=n_pcs, random_state=0).fit_transform(Xz)

# ========= NEW: Build kNN graph explicitly (cosine) and run UMAP from it =========
# Create a tiny AnnData just to leverage Scanpy’s neighbor graph + UMAP
adata = sc.AnnData(Xpca)                       # rows = drugs, columns = PCs
# Build the neighbor graph in this PCA space (cosine usually best for DEGs)
K = 15  # try 15, 30, 50 to see which gives best MoA separation
sc.pp.neighbors(adata, n_neighbors=K, metric="cosine", use_rep="X")

# UMAP on the precomputed neighbor graph; tune min_dist for tighter vs looser clusters
sc.tl.umap(adata, min_dist=0.50, spread=2.0)   # try min_dist in [0.05, 0.5]
emb = adata.obsm["X_umap"]                     # (N, 2) UMAP coordinates

# --- 5) Plot (unchanged styling, but using 'emb' computed from the graph) ---
plt.figure(figsize=(9,7))
label_col = "pert_iname"
labels = df_topoi[label_col].astype(str).values

codes, uniques = pd.factorize(labels)
n_cls = len(uniques)
cmap = mpl.cm.get_cmap('tab20', n_cls) if n_cls > 10 else mpl.cm.get_cmap('tab10', n_cls)
colors = cmap(codes)

scat = plt.scatter(emb[:,0], emb[:,1], c=colors, s=16, alpha=0.85, linewidths=0)

handles = [Line2D([0],[0], marker='o', linestyle='',
                  markersize=6, markerfacecolor=cmap(i), markeredgecolor='none')
           for i in range(n_cls)]
legend_title = "Drug"
plt.legend(handles, uniques, title=legend_title, bbox_to_anchor=(1.02,1),
           loc="upper left", borderaxespad=0., ncol=1, fontsize=9)

plt.title(f"Grouped by Drug")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.tight_layout()

outfile = "./Code/downstream_analysis_code/ExploratoryAnalysis/Figs/PredictedLINCS_umap_nsclc_Topoisomerase_graphfirst_k15_mindist05Spread2_compare.pdf"
plt.savefig(outfile, bbox_inches="tight")
plt.close()

### group by duration
df_topoi = df_pred_nsclc[df_pred_nsclc['moa_group']=='Topoisomerase inhibitor']
meta_cols = [c for c in df_topoi.columns if _norm_colname(c) in META_CANDIDATES]

# take all numeric columns, then drop any that are known metadata by name
num_cols  = df_topoi.select_dtypes(include=[np.number]).columns.tolist()
gene_cols = [c for c in num_cols if _norm_colname(c) not in META_CANDIDATES]
assert len(gene_cols) > 0, "No numeric feature columns detected (after excluding metadata)."

# --- 2) Labels (best available) ---
label_col = 'pert_time'
labels = df_topoi[label_col].astype(str).values if label_col else None

# --- 3) Matrix + scaling (same as before) ---
X = df_topoi[gene_cols].to_numpy(dtype=float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
scaler = StandardScaler(with_mean=True, with_std=True)
Xz = scaler.fit_transform(X)

# --- 4) PCA (same idea; keep ~50 PCs) ---
n_pcs = min(50, Xz.shape[1])
Xpca = PCA(n_components=n_pcs, random_state=0).fit_transform(Xz)

# ========= NEW: Build kNN graph explicitly (cosine) and run UMAP from it =========
# Create a tiny AnnData just to leverage Scanpy’s neighbor graph + UMAP
adata = sc.AnnData(Xpca)                       # rows = drugs, columns = PCs
# Build the neighbor graph in this PCA space (cosine usually best for DEGs)
K = 15  # try 15, 30, 50 to see which gives best MoA separation
sc.pp.neighbors(adata, n_neighbors=K, metric="cosine", use_rep="X")

# UMAP on the precomputed neighbor graph; tune min_dist for tighter vs looser clusters
sc.tl.umap(adata, min_dist=0.50, spread=2.0)   # try min_dist in [0.05, 0.5]
emb = adata.obsm["X_umap"]                     # (N, 2) UMAP coordinates

# --- 5) Plot (unchanged styling, but using 'emb' computed from the graph) ---
plt.figure(figsize=(9,7))
label_col = "pert_time"
labels = df_topoi[label_col].astype(str).values

codes, uniques = pd.factorize(labels)
n_cls = len(uniques)
cmap = mpl.cm.get_cmap('tab20', n_cls) if n_cls > 10 else mpl.cm.get_cmap('tab10', n_cls)
colors = cmap(codes)

scat = plt.scatter(emb[:,0], emb[:,1], c=colors, s=16, alpha=0.85, linewidths=0)

handles = [Line2D([0],[0], marker='o', linestyle='',
                  markersize=6, markerfacecolor=cmap(i), markeredgecolor='none')
           for i in range(n_cls)]
legend_title = "Perturbation Duration"
plt.legend(handles, uniques, title=legend_title, bbox_to_anchor=(1.02,1),
           loc="upper left", borderaxespad=0., ncol=1, fontsize=9)

plt.title(f"Grouped by Perturbation Duration")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.tight_layout()

outfile = "./Code/downstream_analysis_code/ExploratoryAnalysis/Figs/PredictedLINCS_umap_nsclc_TopoisomeraseByTime_graphfirst_k15_mindist05Spread2_compare.pdf"
plt.savefig(outfile, bbox_inches="tight")
plt.close()

### group by dosage
df_topoi = df_pred_nsclc[df_pred_nsclc['moa_group']=='Topoisomerase inhibitor']
meta_cols = [c for c in df_topoi.columns if _norm_colname(c) in META_CANDIDATES]

# take all numeric columns, then drop any that are known metadata by name
num_cols  = df_topoi.select_dtypes(include=[np.number]).columns.tolist()
gene_cols = [c for c in num_cols if _norm_colname(c) not in META_CANDIDATES]
assert len(gene_cols) > 0, "No numeric feature columns detected (after excluding metadata)."

# --- 2) Labels (best available) ---
label_col = 'dose'
labels = df_topoi[label_col].astype(str).values if label_col else None

# --- 3) Matrix + scaling (same as before) ---
X = df_topoi[gene_cols].to_numpy(dtype=float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
scaler = StandardScaler(with_mean=True, with_std=True)
Xz = scaler.fit_transform(X)

# --- 4) PCA (same idea; keep ~50 PCs) ---
n_pcs = min(50, Xz.shape[1])
Xpca = PCA(n_components=n_pcs, random_state=0).fit_transform(Xz)

# ========= NEW: Build kNN graph explicitly (cosine) and run UMAP from it =========
# Create a tiny AnnData just to leverage Scanpy’s neighbor graph + UMAP
adata = sc.AnnData(Xpca)                       # rows = drugs, columns = PCs
# Build the neighbor graph in this PCA space (cosine usually best for DEGs)
K = 15  # try 15, 30, 50 to see which gives best MoA separation
sc.pp.neighbors(adata, n_neighbors=K, metric="cosine", use_rep="X")

# UMAP on the precomputed neighbor graph; tune min_dist for tighter vs looser clusters
sc.tl.umap(adata, min_dist=0.50, spread=2.0)   # try min_dist in [0.05, 0.5]
emb = adata.obsm["X_umap"]                     # (N, 2) UMAP coordinates

# --- 5) Plot with continuous colormap by numeric dose ---
dose = pd.to_numeric(df_topoi['dose'], errors='coerce')

# optional: clip extremes to avoid a few outliers dominating the scale
vmin, vmax = np.nanpercentile(dose, [2, 98])

# if doses are >0 and span orders of magnitude, LogNorm is nice; otherwise use Normalize
use_log = np.isfinite(dose).all() and (dose.min() > 0) and (dose.max() / dose.min() >= 50)
norm = mpl.colors.LogNorm(vmin=max(vmin, np.finfo(float).eps), vmax=vmax) if use_log else mpl.colors.Normalize(vmin=vmin, vmax=vmax)

plt.figure(figsize=(9,7))
scat = plt.scatter(emb[:,0], emb[:,1],
                   c=dose, cmap='viridis', norm=norm,
                   s=16, alpha=0.85, linewidths=0)

# mark NaN doses (if any) in gray on top
nan_mask = dose.isna().values
if nan_mask.any():
    plt.scatter(emb[nan_mask,0], emb[nan_mask,1], c='lightgray', s=16, alpha=0.85, linewidths=0, label='dose NA')
    plt.legend(loc='upper left')

cbar = plt.colorbar(scat, shrink=0.9)
cbar.set_label('Dose')

plt.title(f"Grouped by Drug Dosage")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")

plt.tight_layout()

outfile = "./Code/downstream_analysis_code/ExploratoryAnalysis/Figs/PredictedLINCS_umap_nsclc_TopoisomeraseByDose_graphfirst_k15_mindist05Spread2_compare.pdf"
plt.savefig(outfile, bbox_inches="tight")
plt.close()


### assign topo group
# 1) Normalizer to make matching robust (case, spaces, various hyphens, SN38 vs SN-38)
def _norm_drug(x):
    if pd.isna(x):
        return ""
    s = str(x).lower().strip()
    s = re.sub(r'[\u2010-\u2015–—−]', '-', s)  # unify all dash types to '-'
    s = s.replace(' ', '')                     # drop spaces
    return s

# 2) Define Type I / II sets (write them once in their normal forms; the code normalizes)
type_I_raw = {
    "10-hydroxycamptothecin", "sn-38", "camptothecin", "irinotecan"
}
type_II_raw = {
    "idarubicin","amonafide","amsacrine","daunorubicin","dexrazoxane",
    "doxorubicin","epirubicin","etoposide","mitoxantrone",
    "pirarubicin","teniposide","topotecan"
}

### group by duration
df_topoi = df_pred_nsclc[df_pred_nsclc['moa_group']=='Topoisomerase inhibitor']
meta_cols = [c for c in df_topoi.columns if _norm_colname(c) in META_CANDIDATES]

# take all numeric columns, then drop any that are known metadata by name
num_cols  = df_topoi.select_dtypes(include=[np.number]).columns.tolist()
gene_cols = [c for c in num_cols if _norm_colname(c) not in META_CANDIDATES]
assert len(gene_cols) > 0, "No numeric feature columns detected (after excluding metadata)."

# --- 2) Labels (best available) ---
label_col = 'pert_time'
labels = df_topoi[label_col].astype(str).values if label_col else None

# --- 3) Matrix + scaling (same as before) ---
X = df_topoi[gene_cols].to_numpy(dtype=float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
scaler = StandardScaler(with_mean=True, with_std=True)
Xz = scaler.fit_transform(X)

# --- 4) PCA (same idea; keep ~50 PCs) ---
n_pcs = min(50, Xz.shape[1])
Xpca = PCA(n_components=n_pcs, random_state=0).fit_transform(Xz)

# ========= NEW: Build kNN graph explicitly (cosine) and run UMAP from it =========
# Create a tiny AnnData just to leverage Scanpy’s neighbor graph + UMAP
adata = sc.AnnData(Xpca)                       # rows = drugs, columns = PCs
# Build the neighbor graph in this PCA space (cosine usually best for DEGs)
K = 15  # try 15, 30, 50 to see which gives best MoA separation
sc.pp.neighbors(adata, n_neighbors=K, metric="cosine", use_rep="X")

# UMAP on the precomputed neighbor graph; tune min_dist for tighter vs looser clusters
sc.tl.umap(adata, min_dist=0.50, spread=2.0)   # try min_dist in [0.05, 0.5]
emb = adata.obsm["X_umap"]                     # (N, 2) UMAP coordinates

# --- 5) Plot (unchanged styling, but using 'emb' computed from the graph) ---
plt.figure(figsize=(9,7))
label_col = "pert_time"
labels = df_topoi[label_col].astype(str).values

codes, uniques = pd.factorize(labels)
n_cls = len(uniques)
cmap = mpl.cm.get_cmap('tab20', n_cls) if n_cls > 10 else mpl.cm.get_cmap('tab10', n_cls)
colors = cmap(codes)

scat = plt.scatter(emb[:,0], emb[:,1], c=colors, s=16, alpha=0.85, linewidths=0)

handles = [Line2D([0],[0], marker='o', linestyle='',
                  markersize=6, markerfacecolor=cmap(i), markeredgecolor='none')
           for i in range(n_cls)]
plt.legend(handles, uniques, title=label_col, bbox_to_anchor=(1.02,1),
           loc="upper left", borderaxespad=0., ncol=1, fontsize=9)

plt.title(f"UMAP (cosine kNN={K}) of Predicted Signatures")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.tight_layout()

outfile = "./Code/downstream_analysis_code/ExploratoryAnalysis/Figs/PredictedLINCS_umap_nsclc_TopoisomeraseByTime_graphfirst_k15_mindist05Spread2_compare.pdf"
plt.savefig(outfile, bbox_inches="tight")
plt.close()

# Normalize the lookup sets
type_I = {_norm_drug(x) for x in type_I_raw}
type_II = {_norm_drug(x) for x in type_II_raw}

# 3) Compute normalized names from df_topoi and assign topo_group
norm_names = df_topoi['pert_iname'].apply(_norm_drug)

df_topoi['topo_group'] = np.where(
    norm_names.isin(type_I), 'Type I',
    np.where(norm_names.isin(type_II), 'Type II', pd.NA)
)

# take all numeric columns, then drop any that are known metadata by name
num_cols  = df_topoi.select_dtypes(include=[np.number]).columns.tolist()
gene_cols = [c for c in num_cols if _norm_colname(c) not in META_CANDIDATES]
assert len(gene_cols) > 0, "No numeric feature columns detected (after excluding metadata)."

# --- 2) Labels (best available) ---
label_col = 'topo_group'
labels = df_topoi[label_col].astype(str).values if label_col else None

# --- 3) Matrix + scaling (same as before) ---
X = df_topoi[gene_cols].to_numpy(dtype=float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
scaler = StandardScaler(with_mean=True, with_std=True)
Xz = scaler.fit_transform(X)

# --- 4) PCA (same idea; keep ~50 PCs) ---
n_pcs = min(50, Xz.shape[1])
Xpca = PCA(n_components=n_pcs, random_state=0).fit_transform(Xz)

# ========= NEW: Build kNN graph explicitly (cosine) and run UMAP from it =========
# Create a tiny AnnData just to leverage Scanpy’s neighbor graph + UMAP
adata = sc.AnnData(Xpca)                       # rows = drugs, columns = PCs
# Build the neighbor graph in this PCA space (cosine usually best for DEGs)
K = 15  # try 15, 30, 50 to see which gives best MoA separation
sc.pp.neighbors(adata, n_neighbors=K, metric="cosine", use_rep="X")

# UMAP on the precomputed neighbor graph; tune min_dist for tighter vs looser clusters
sc.tl.umap(adata, min_dist=0.50, spread=2.0)   # try min_dist in [0.05, 0.5]
emb = adata.obsm["X_umap"]                     # (N, 2) UMAP coordinates

# --- 5) Plot (unchanged styling, but using 'emb' computed from the graph) ---
plt.figure(figsize=(9,7))
label_col = "topo_group"
labels = df_topoi[label_col].astype(str).values

codes, uniques = pd.factorize(labels)
n_cls = len(uniques)
cmap = mpl.cm.get_cmap('tab20', n_cls) if n_cls > 10 else mpl.cm.get_cmap('tab10', n_cls)
colors = cmap(codes)

scat = plt.scatter(emb[:,0], emb[:,1], c=colors, s=16, alpha=0.85, linewidths=0)

handles = [Line2D([0],[0], marker='o', linestyle='',
                  markersize=6, markerfacecolor=cmap(i), markeredgecolor='none')
           for i in range(n_cls)]
legend_title = "Topoisomerase Inhibitor Type"
plt.legend(handles, uniques, title=legend_title, bbox_to_anchor=(1.02,1),
           loc="upper left", borderaxespad=0., ncol=1, fontsize=9)

plt.title(f"Grouped by Topoisomerase Inhibitor Type")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.tight_layout()

outfile = "./Code/downstream_analysis_code/ExploratoryAnalysis/Figs/PredictedLINCS_umap_nsclc_TopoisomeraseByGroup_graphfirst_k15_mindist05Spread2_compare.pdf"
plt.savefig(outfile, bbox_inches="tight")
plt.close()


'''
inspect outliers
'''
# 1) attach UMAP to the dataframe
df_topoi_outlier = df_topoi.copy()
df_topoi_outlier["umap1"] = emb[:, 0]
df_topoi_outlier["umap2"] = emb[:, 1]

# 2) define your “red circle” box (tweak as needed)
mask1 = df_topoi_outlier["umap2"] < -5
mask2 = df_topoi_outlier["umap2"] > 10

# (optional) restrict to the Topoisomerase group only
# mask = mask & df["moa_group"].astype(str).eq("Topoisomerase inhibitor")

# 3) inspect the hits
cols = ["pert_iname","dose","pert_time","moa_group","moa_nsclc","moa","umap1","umap2"]
hits_underNeg5 = df_topoi_outlier.loc[mask1, cols].sort_values(["umap1","umap2"])
hits_abovePos10 = df_topoi_outlier.loc[mask2, cols].sort_values(["umap1","umap2"])

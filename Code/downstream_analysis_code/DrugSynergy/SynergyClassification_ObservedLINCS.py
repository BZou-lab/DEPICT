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

merged_adata_read = sc.read("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_22drugs_wControl.h5ad")

merged_adata_read.obs['pert_iname'].nunique() # 23

merged_adata_read.obs['pert_iname'].value_counts()
merged_adata_read.obs['control'].value_counts()

'''
step 1: calculate one single drug signature for each drug at one specific duration and dosage.
'''
def compute_pooled_signatures(merged_adata):
    """
    Returns:
        signatures_df: DataFrame with columns:
            ['pert_iname','pert_dose','pert_time', <978 gene columns>]
          (pooled plate-centered mean per (pert_iname, pert_dose, pert_time);
           DMSO and control rows are excluded)
        counts_df: DataFrame with index (pert_iname, pert_dose, pert_time) and column 'n_replicates'
    """
    # --- Unpack matrix and obs ---
    expr_df = merged_adata.to_df()                 # n_obs x n_genes
    obs = merged_adata.obs.copy()

    # Normalize pert_iname for robust DMSO detection (case/whitespace)
    pert_norm = obs['pert_iname'].astype(str).str.strip().str.upper()
    is_dmso = pert_norm.eq('DMSO')

    # Robust control mask (accepts 1/0 or bool) + DMSO as control
    ctrl_col = obs['control']
    if ctrl_col.dtype == bool:
        is_flag_control = ctrl_col.values
    else:
        is_flag_control = pd.to_numeric(ctrl_col, errors='coerce').fillna(0).astype(int).eq(1).values

    mask_ctrl = np.logical_or(is_flag_control, is_dmso.values)

    # --- 1) Plate-wise control means ---
    control_means_by_plate = (
        expr_df[mask_ctrl]
        .groupby(obs.loc[mask_ctrl, 'plate'], observed=True)
        .mean()
    )

    # --- 2) Plate-center every experiment: ΔX = X - mean_control(plate) ---
    plate_series = obs['plate']
    ctrl_means_aligned = control_means_by_plate.reindex(plate_series).to_numpy()
    delta_df = pd.DataFrame(
        expr_df.to_numpy() - ctrl_means_aligned,
        index=expr_df.index,
        columns=expr_df.columns
    )

    # --- 3) Keep ONLY non-control & non-DMSO rows for signatures ---
    mask_treat = ~mask_ctrl
    delta_treat = delta_df[mask_treat]

    # --- 4) Pooled one-step averaging by (drug, dose, time) ---
    g_pert_iname = obs.loc[mask_treat, 'pert_iname']
    g_dose       = obs.loc[mask_treat, 'pert_dose']
    g_time       = obs.loc[mask_treat, 'pert_time']

    grouped = delta_treat.groupby(
        [g_pert_iname, g_dose, g_time],
        dropna=False,
        observed=True,
    )

    signatures_wide = grouped.mean()
    signatures_wide.index.set_names(['pert_iname', 'pert_dose', 'pert_time'], inplace=True)

    # replicate counts (QC)
    counts_df = grouped.size().to_frame(name='n_replicates')

    # --- 5) Add metadata columns to signatures_df and move them to the front ---
    signatures_df = signatures_wide.reset_index()  # adds the three meta columns
    meta_cols = ['pert_iname', 'pert_dose', 'pert_time']
    gene_cols = [c for c in signatures_df.columns if c not in meta_cols]
    signatures_df = signatures_df[meta_cols + gene_cols]

    return signatures_df, counts_df


# Run it
signatures_df, counts_df = compute_pooled_signatures(merged_adata)

'''
sanity check
'''
signatures_df['pert_iname'].nunique() # 22 drugs

'''
now match the closest dosage with reference and longest duration
'''
reference_meta = df_diffexp_pred_read[['pert_iname', 'dose', 'pert_time']].copy()

def match_reference_to_signatures(reference_meta: pd.DataFrame, signatures_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols_sig = {'pert_iname', 'pert_dose', 'pert_time', 'n_replicates'}
    gene_cols = [c for c in signatures_df.columns if c not in meta_cols_sig]

    ref = reference_meta.copy()
    ref['dose'] = pd.to_numeric(ref['dose'], errors='coerce')
    ref['pert_time'] = pd.to_numeric(ref['pert_time'], errors='coerce')

    sig = signatures_df.copy()
    sig['pert_dose'] = pd.to_numeric(sig['pert_dose'], errors='coerce')
    sig['pert_time'] = pd.to_numeric(sig['pert_time'], errors='coerce')

    # Add observed=True to silence FutureWarning
    sig_by_drug = {d: g.reset_index(drop=True)
                   for d, g in sig.groupby('pert_iname', dropna=False, observed=True)}

    out_rows = []
    for _, row in ref.iterrows():
        drug = row['pert_iname']
        dose_ref = row['dose']
        time_ref = row['pert_time']

        g = sig_by_drug.get(drug)
        if g is None or g.empty or pd.isna(dose_ref):
            out = {
                'pert_iname_ref': drug,
                'dose_ref': dose_ref,
                'pert_time_ref': time_ref,
                'pert_iname_matched': np.nan,
                'pert_dose_matched': np.nan,
                'pert_time_matched': np.nan,
            }
            out.update({c: np.nan for c in gene_cols})
            out_rows.append(out)
            continue

        dose_diff = (g['pert_dose'] - dose_ref).abs()
        min_diff = dose_diff.min()
        candidates = g.loc[dose_diff == min_diff]

        max_time = candidates['pert_time'].max()
        chosen = candidates.loc[candidates['pert_time'] == max_time].iloc[0]

        out = {
            'pert_iname_ref': drug,
            'dose_ref': dose_ref,
            'pert_time_ref': time_ref,
            'pert_iname_matched': chosen['pert_iname'],
            'pert_dose_matched': chosen['pert_dose'],
            'pert_time_matched': chosen['pert_time'],
        }
        out.update({c: chosen[c] for c in gene_cols})
        out_rows.append(out)

    matched_df = pd.DataFrame(out_rows)
    front = ['pert_iname_ref', 'dose_ref', 'pert_time_ref',
             'pert_iname_matched', 'pert_dose_matched', 'pert_time_matched']
    return matched_df[front + [c for c in matched_df.columns if c not in front]]

# Example usage:
matched_df = match_reference_to_signatures(reference_meta, signatures_df)
matched_df.head()



'''
now the synergy classification code
steps:
    1. PCA into 50 dimension for every drug under each dosage regime.
    2. concat the 50D for each drug pair into 100D, do the prediction
    3. see AUC, ACC, macro F1 and the confusion matrix.
'''
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, average_precision_score

'''
make sure the drug names are consistent
'''
name_map = {d.lower(): d for d in matched_df["pert_iname_ref"].unique()}

# Apply mapping to drug1 and drug2
drug_doublet_label["drug1"] = drug_doublet_label["drug1"].str.lower().map(name_map)
drug_doublet_label["drug2"] = drug_doublet_label["drug2"].str.lower().map(name_map)

drugs_ht29 = pd.unique(
    pd.concat([drug_doublet_label['drug1'], drug_doublet_label['drug2']])
)
drugs_lincs = matched_df['pert_iname_ref']
common_drugs = set(drugs_lincs).intersection(set(drugs_ht29))

# --------------------------
# Step 1. PCA on transcriptional data
# --------------------------

# Assume df_diffexp_read and drug_doublet_label are already loaded

# Get only gene expression columns (first 978)
gene_cols = matched_df.columns[6:]
X_gene = matched_df[gene_cols].values

# PCA to 50D
pca = PCA(n_components=50, random_state=6666)
X_pca = pca.fit_transform(X_gene)

# Store PCA features back
df_pca = pd.DataFrame(X_pca, index=matched_df.index,
                      columns=[f"PC{i + 1}" for i in range(50)])
df_pca["pert_iname"] = matched_df["pert_iname_ref"]
df_pca["dose"] = matched_df["dose_ref"]

# --------------------------
# Step 2. Forge dataset
# --------------------------

features = []
labels = []

for _, row in drug_doublet_label.iterrows():
    drug1, drug2 = row["drug1"], row["drug2"]
    d1, d2 = row["drug1_conc"], row["drug2_conc"]

    # Find PCA vector for drug1 at given dose
    f1 = df_pca[(df_pca["pert_iname"] == drug1) &
                (df_pca["dose"] == d1)].iloc[0, :-2].values

    # Find PCA vector for drug2 at given dose
    f2 = df_pca[(df_pca["pert_iname"] == drug2) &
                (df_pca["dose"] == d2)].iloc[0, :-2].values

    # Concatenate into 100D
    feat = np.concatenate([f1, f2])
    features.append(feat)

    labels.append(row["label"])

X = np.array(features)
y = np.array(labels)

# Encode labels as binary (synergy=1, antagonism=0)
# If you still have "additive", drop them earlier
y_binary = np.where(y == "synergy", 1, 0)

# --------------------------
# Step 3. Leave-One-Out CV
# --------------------------
loo = LeaveOneOut()

preds_lr, probs_lr = [], []
preds_rf, probs_rf = [], []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_binary[train_idx], y_binary[test_idx]

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, solver="liblinear")
    lr.fit(X_train, y_train)
    probs = lr.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    probs_lr.extend(probs)
    preds_lr.extend(preds)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    probs_rf.extend(probs)
    preds_rf.extend(preds)


# --------------------------
# Step 4. Evaluation
# --------------------------

def evaluate(y_true, preds, probs, model_name):
    auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)  # PR-AUC (area under Precision–Recall)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    cm = confusion_matrix(y_true, preds)
    print(f"\n=== {model_name} ===")
    print(f"AUC (ROC): {auc:.3f}")
    print(f"PR-AUC:     {pr_auc:.3f}")
    print(f"Accuracy:   {acc:.3f}")
    print(f"Macro F1:   {f1:.3f}")
    print("Confusion matrix:\n", cm)


evaluate(y_binary, preds_lr, probs_lr, "Logistic Regression")
evaluate(y_binary, preds_rf, probs_rf, "Random Forest")

'''
results

=== Logistic Regression ===
AUC (ROC): 0.786
PR-AUC:     0.687
Accuracy:   0.752
Macro F1:   0.717
Confusion matrix:
 [[306  51]
 [ 87 112]]
 
=== Random Forest ===
AUC (ROC): 0.777
PR-AUC:     0.639
Accuracy:   0.745
Macro F1:   0.714
Confusion matrix:
 [[298  59]
 [ 83 116]]

'''

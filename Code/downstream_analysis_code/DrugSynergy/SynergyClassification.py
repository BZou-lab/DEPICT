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
# sanity check read in
df_diffexp_read = pd.read_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/diff_geneExp_pred_ht29.csv", index_col=0) # all good

'''
now read in the drug synergy label based on Loewe additivity score.
'''
drug_doublet_label = pd.read_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_allpairs_LoeweCI_labels.csv")

'''
make sure the drug names are consistent
'''
name_map = {d.lower(): d for d in df_diffexp_read["pert_iname"].unique()}

# Apply mapping to drug1 and drug2
drug_doublet_label["drug1"] = drug_doublet_label["drug1"].str.lower().map(name_map)
drug_doublet_label["drug2"] = drug_doublet_label["drug2"].str.lower().map(name_map)

drugs_ht29 = pd.unique(
    pd.concat([drug_doublet_label['drug1'], drug_doublet_label['drug2']])
)
drugs_lincs = df_diffexp_read['pert_iname']
common_drugs = set(drugs_lincs).intersection(set(drugs_ht29))

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

# --------------------------
# Step 1. PCA on transcriptional data
# --------------------------

# Assume df_diffexp_read and drug_doublet_label are already loaded

# Get only gene expression columns (first 978)
gene_cols = df_diffexp_read.columns[:978]
X_gene = df_diffexp_read[gene_cols].values

# PCA to 50D
pca = PCA(n_components=50, random_state=6666)
X_pca = pca.fit_transform(X_gene)

# Store PCA features back
df_pca = pd.DataFrame(X_pca, index=df_diffexp_read.index,
                      columns=[f"PC{i + 1}" for i in range(50)])
df_pca["pert_iname"] = df_diffexp_read["pert_iname"]
df_pca["dose"] = df_diffexp_read["dose"]

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
evaluate(y_binary, preds_lr, probs_lr, "Logistic Regression")
=== Logistic Regression ===
AUC (ROC): 0.844
PR-AUC:     0.714
Accuracy:   0.802
Macro F1:   0.784
Confusion matrix:
 [[304  53]
 [ 57 142]]
evaluate(y_binary, preds_rf, probs_rf, "Random Forest")
=== Random Forest ===
AUC (ROC): 0.855
PR-AUC:     0.770
Accuracy:   0.802
Macro F1:   0.782
Confusion matrix:
 [[308  49]
 [ 61 138]]
'''

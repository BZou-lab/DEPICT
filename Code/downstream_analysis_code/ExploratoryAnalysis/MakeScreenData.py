import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import scipy.sparse as sp
import warnings

# this dataset should be created in the Drug Prioritization analysis.
adata_a549 = sc.read('./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/a549.h5ad')

drug_nsclc = pd.read_csv("./Code/downstream_analysis_code/ExploratoryAnalysis/Data/DrugsOfInterestMoA.csv",index_col=0)

def compute_platewise_dge_df(
    adata,
    control_col="control",
    plate_col="plate",
    meta_cols=("pert_iname", "dose", "pert_time"),
    fallback_to_global_ctrl=True,  # kept True so the pipeline doesn't break; we also WARN loudly
):
    """
    Compute differential gene expression per treatment:
      DGE(treatment) = X_treatment - mean_control_expression_on_same_plate

    Parameters
    ----------
    adata : AnnData
        AnnData with .X shape (n_cells/experiments, n_genes) and required obs columns.
    control_col : str
        obs column indicating controls (1) vs treatments (0).
    plate_col : str
        obs column (string) denoting plate ID.
    meta_cols : tuple[str]
        Metadata columns to append to the right of the 978 gene columns.
    fallback_to_global_ctrl : bool
        If True and a treatment plate has no controls, use the global control mean
        *and emit a warning*. This should not trigger if your data are consistent.

    Returns
    -------
    pd.DataFrame
        Rows = treatment experiments (obs_names as index).
        First G columns (G = number of genes, e.g. 978) = DGE, in adata.var_names order.
        Then the specified meta columns.
    """
    # --- Basic checks
    for c in (control_col, plate_col, *meta_cols):
        if c not in adata.obs:
            raise KeyError(f"obs['{c}'] not found in AnnData.")

    # --- Get X as dense numpy array (handle sparse)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    # --- Masks and keys
    control_mask = (adata.obs[control_col].astype(int).values == 1)
    treat_mask   = ~control_mask

    if control_mask.sum() == 0:
        raise ValueError("No controls found (obs['control'] == 1). Cannot compute baselines.")

    plates = adata.obs[plate_col].astype(str)
    var_names = list(map(str, adata.var_names))

    # --- Per-plate control means
    plate_control_means = {}
    for p in plates[control_mask].unique():
        rows = (plates == p).values & control_mask
        plate_control_means[p] = X[rows].mean(axis=0)

    # --- Global control mean (for rare fallback)
    global_ctrl_mean = X[control_mask].mean(axis=0)

    # --- Build baseline matrix aligned to treatment rows
    treat_plates = plates[treat_mask].astype(str).values
    missing_plates = sorted(set(treat_plates) - set(plate_control_means.keys()))
    if missing_plates:
        msg = (
            "Sanity check: The following treatment plates have no controls in obs "
            f"and will use the GLOBAL control mean as fallback: {missing_plates}"
        )
        warnings.warn(msg)

    if fallback_to_global_ctrl:
        baseline_rows = np.vstack([plate_control_means.get(p, global_ctrl_mean) for p in treat_plates])
    else:
        if missing_plates:
            raise ValueError(
                "Missing per-plate controls for: " + ", ".join(missing_plates)
            )
        baseline_rows = np.vstack([plate_control_means[p] for p in treat_plates])

    # --- Compute DGE for treatments
    X_treat = X[treat_mask]
    dge = X_treat - baseline_rows

    # --- Assemble DataFrame: genes first (in adata.var_names order), then metadata
    df_genes = pd.DataFrame(dge, columns=var_names, index=adata.obs_names[treat_mask])
    df_meta  = adata.obs.loc[treat_mask, list(meta_cols)]
    df_dge   = pd.concat([df_genes, df_meta], axis=1)

    return df_dge

df_dge = compute_platewise_dge_df(adata_a549)

'''
filter out the drugs with full MoA and targets information
'''
# 1. Clean drug_repur: drop rows missing moa or target
drug_repur_clean = drug_nsclc.dropna(subset=["moa", "target"]).copy()

# 2. Build the allowed drug set (lowercased)
allowed_drugs = (
    drug_repur_clean["pert_iname"]
    .astype(str)
    .str.strip()
    .str.lower()
    .unique()
)
allowed_drugs = set(allowed_drugs)

# 3. Make a lowercase version of pert_iname in df_dge for matching
df_dge_work = df_dge.copy()
df_dge_work["pert_iname_lc"] = (
    df_dge_work["pert_iname"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# 4. Filter df_dge by membership in allowed set
mask = df_dge_work["pert_iname_lc"].isin(allowed_drugs)
df_dge_repur = df_dge_work.loc[mask].drop(columns=["pert_iname_lc"])

# (Optional) sanity checks / quick summary
print(f"Original df_dge rows: {len(df_dge)}")
print(f"After filtering using drug_repur (with moa+target): {len(df_dge_repur)}")
print(f"Unique pert_iname kept: {df_dge_repur['pert_iname'].nunique()} "
      f"out of {df_dge['pert_iname'].nunique()} original") # 166 unique drugs

'''
now deal with the replicates(same drug, duration and dosage) in the df_dge_nsclc.
'''
# Columns to keep as identifiers
# --- FIX: build df_dge_nsclc_avg with the right key order ---
meta_cols = ["pert_iname", "dose", "pert_time"]
gene_cols = [c for c in df_dge_repur.columns if c not in meta_cols]

df_work = df_dge_repur.copy()
df_work[gene_cols] = df_work[gene_cols].apply(pd.to_numeric, errors="coerce")

key_pert = df_work["pert_iname"].astype(str).str.strip()
key_dose = df_work["dose"]        # <-- dose second
key_time = df_work["pert_time"]   # <-- time third

df_avg = (
    df_work
    .groupby([key_pert, key_dose, key_time], dropna=False)[gene_cols]  # <-- order: pert, dose, time
    .mean()
    .reset_index(names=["pert_iname", "dose", "pert_time"])            # <-- names match that order
)

df_dge_repur_avg = df_avg[gene_cols + meta_cols]

# Optional: quick summary
n_before = len(df_dge_repur)
n_after = len(df_dge_repur_avg)
print(f"Averaged replicates: {n_before} → {n_after} unique (pert_iname, pert_time, dose) combos.")

df_dge_repur_avg['pert_iname'].nunique() # 166 unique drugs with 1110 combinations, with known MoA and drug targets


'''
now merge the MoAs with the df_dge_repur_avg.
'''
# Columns we want to bring over from drug_nsclc
anno_cols = ["moa", "target", "moa_nsclc", "moa_group"]

# 1. Sanity checks
if "pert_iname" not in df_dge_repur_avg.columns:
    raise KeyError("df_dge_repur_avg must contain 'pert_iname'.")
if "pert_iname" not in drug_nsclc.columns:
    raise KeyError("drug_nsclc must contain 'pert_iname'.")
missing_cols = [c for c in anno_cols if c not in drug_nsclc.columns]
if missing_cols:
    raise KeyError(f"drug_nsclc is missing columns: {missing_cols}")

# 2. Create lowercase merge keys
df_left = df_dge_repur_avg.copy()
df_left["pert_iname_key"] = (
    df_left["pert_iname"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df_right = drug_nsclc.copy()
df_right["pert_iname_key"] = (
    df_right["pert_iname"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# 3. Deduplicate df_right so each pert_iname_key appears once
#    Strategy: take the first observed row per key.
#    (If you'd rather combine multiple rows per drug, we can replace this with a groupby + join.)
df_right_dedup = (
    df_right[["pert_iname_key"] + anno_cols]
    .drop_duplicates(subset=["pert_iname_key"])
)

# 4. Merge on the normalized key
df_dge_repur_avg_annot = df_left.merge(
    df_right_dedup,
    on="pert_iname_key",
    how="left",
)

# 5. Drop helper key column and restore final column order
df_dge_repur_avg_annot = df_dge_repur_avg_annot.drop(columns=["pert_iname_key"])

# 6. Sanity check again
missing_after = (
    df_dge_repur_avg_annot["moa"].isna().any(),
    df_dge_repur_avg_annot["target"].isna().any()
)
print("Any drugs with missing annotation after lowercase merge?", missing_after)

'''
make screened data
'''
def build_screen_from_df(
    adata_a549: AnnData,
    df_source: pd.DataFrame,
    cell_id_value: str = "A549",
) -> AnnData:
    """
    Build screening AnnData from unique (pert_iname, dose, pert_time) triplets in df_source.

    For each unique triplet:
      1. Find all matching rows in adata_a549.obs (same pert_iname, dose, pert_time).
      2. Get the unique plates for those rows.
      3. From those plates, gather all control rows (control == 1).
      4. Average those control rows' expression profiles to form one baseline vector.
      5. Store that vector as one row in .X.

    Output AnnData:
      - X: stacked baseline vectors (float32)
      - obs: columns ['cell_id','pert_iname','dose','pert_time','n_ctrls']
        index: "pert_iname|dose|pert_time"
      - var: copied from adata_a549.var
    """

    # required columns in adata_a549
    required_obs_cols = ["pert_iname", "dose", "pert_time", "plate", "control"]
    for col in required_obs_cols:
        if col not in adata_a549.obs.columns:
            raise KeyError(f"adata_a549.obs must contain '{col}'.")

    # required columns in df_source
    for col in ["pert_iname", "dose", "pert_time"]:
        if col not in df_source.columns:
            raise KeyError(f"df_source must contain '{col}'.")

    obs_full = adata_a549.obs
    X_all = adata_a549.X

    # map obs index → row index in X
    row_pos = pd.Series(np.arange(obs_full.shape[0]), index=obs_full.index)

    # helper to average control expression across a set of obs row indices
    def mean_over_rows(idxs: np.ndarray) -> np.ndarray:
        if idxs.size == 0:
            raise ValueError("mean_over_rows() called with empty idxs.")
        if sp.issparse(X_all):
            v = X_all[idxs, :].mean(axis=0)
            return np.asarray(v).ravel().astype(np.float32)
        return (
            np.asarray(X_all[idxs, :], dtype=np.float32)
            .mean(axis=0, dtype=np.float32)
        )

    # unique triplets from df_source (these define the screen conditions)
    triplets = (
        df_source[["pert_iname", "dose", "pert_time"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    X_rows = []
    obs_rows = []
    skipped = []

    for (drug, dose_val, time_val) in triplets:
        # 1. find matching rows in adata_a549 with same triplet
        m_trip = (
            (obs_full["pert_iname"].values == drug) &
            (obs_full["dose"].values == dose_val) &
            (obs_full["pert_time"].values == time_val)
        )

        if not m_trip.any():
            skipped.append((drug, dose_val, time_val, "no matching rows in adata_a549"))
            continue

        # 2. collect unique plates for those matches
        plates = pd.unique(obs_full.loc[m_trip, "plate"])
        if plates.size == 0:
            skipped.append((drug, dose_val, time_val, "matched rows have no plate info"))
            continue

        # 3. controls on those plates
        is_ctrl = obs_full["control"].astype(int).values == 1
        same_plate_as_treatment = obs_full["plate"].isin(plates).values
        m_ctrl = is_ctrl & same_plate_as_treatment

        if not m_ctrl.any():
            skipped.append((drug, dose_val, time_val, "no controls on those plates"))
            continue

        ctrl_row_idxs = row_pos.loc[obs_full.index[m_ctrl]].to_numpy()

        # 4. baseline = mean control expression across these control wells
        baseline_vec = mean_over_rows(ctrl_row_idxs)

        # 5. record
        X_rows.append(baseline_vec)
        obs_rows.append({
            "cell_id": cell_id_value,
            "pert_iname": drug,
            "dose": dose_val,
            "pert_time": time_val,
            "n_ctrls": int(m_ctrl.sum()),
        })

    # if nothing matched at all, raise with useful info
    if not X_rows:
        raise ValueError(
            "No screening rows could be built. "
            + "; ".join([f"{d}|{dose}|{t}: {why}" for d, dose, t, why in skipped])
        )

    # stack into final X
    X_new = np.vstack(X_rows).astype(np.float32)
    obs_new = pd.DataFrame(obs_rows)

    # make index "pert_iname|dose|pert_time"
    obs_index = (
        obs_new["pert_iname"].astype(str)
        + "|" + obs_new["dose"].astype(str)
        + "|" + obs_new["pert_time"].astype(str)
    )
    obs_new.index = pd.Index(obs_index, name="obs_name")

    # build AnnData
    adata_screen = AnnData(
        X=X_new,
        obs=obs_new,
        var=adata_a549.var.copy()
    )

    # warn if some triplets couldn't be built (e.g. missing controls etc.)
    if skipped:
        warnings.warn(
            "Some conditions skipped: "
            + "; ".join([f"{d}|{dose}|{t} -> {why}" for d, dose, t, why in skipped])
        )

    return adata_screen


adata_screen = build_screen_from_df(adata_a549, df_dge_repur_avg_annot)
adata_screen.write("./Code/downstream_analysis_code/ExploratoryAnalysis/Data/a549_166drugMoA.h5ad")

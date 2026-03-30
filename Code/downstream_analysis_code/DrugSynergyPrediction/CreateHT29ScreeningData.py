import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

adata_ht29 = sc.read("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/ht29.h5ad")
drug_single = pd.read_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_singleData.csv", index_col=0)
drug_comb = pd.read_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_pairData.csv", index_col=0)

'''
check the drug name is consistent or not for LINCS and the reference dataset
'''
drugs_adata = adata_ht29.obs['pert_iname'].unique()
drugs_ht29_single = drug_single['drug_name'].unique()
common_drugs_ht29 = set(drugs_ht29_single).intersection(set(drugs_adata)) # 4

drugs_adata = adata_ht29.obs['pert_iname'].str.lower().unique()
drugs_ht29_single = drug_single['drug_name'].str.lower().unique()
common_drugs_ht29 = set(drugs_ht29_single).intersection(set(drugs_adata)) # 22
'''
name not consistent, change drugs names into the LINCS's format
'''
# Step 1: collect the drug names from adata
adata_drugs = set(adata_ht29.obs['pert_iname'].unique())

# build a mapping dictionary using lowercase as the key
drug_map = {d.lower(): d for d in adata_drugs}

# Step 2: check what drugs exist in your single and combo dfs
single_drugs = set(drug_single['drug_name'].str.lower())
comb_drugs = set(drug_comb['drugA_name'].str.lower()).union(
    set(drug_comb['drugB_name'].str.lower())
)

# common drugs across all datasets (case-insensitive)
common_drugs = (set(drug_map.keys()) & single_drugs & comb_drugs)
print(f"Found {len(common_drugs)} common drugs")

# Step 3: replace in both dataframes
drug_single['drug_name'] = drug_single['drug_name'].str.lower().map(drug_map)
drug_comb['drugA_name'] = drug_comb['drugA_name'].str.lower().map(drug_map)
drug_comb['drugB_name'] = drug_comb['drugB_name'].str.lower().map(drug_map)


'''
check the HT29 duration
'''
duration_counts = adata_ht29.obs['pert_time'].value_counts()
'''
bad news is the LINCS only have 6 or 24-hour duration for HT29; However, the reference data is 96-hour duration for HT29.
Use 24 hour duration as control input.
'''
adata_ht29_24h = adata_ht29[adata_ht29.obs['pert_time']==24].copy()

'''
first extract all the dose regime for every drug
'''
# From single-drug table
df_single = drug_single[['drug_name', 'Drug_concentration (µM)']].copy()
df_single = df_single.rename(columns={'drug_name': 'pert_iname',
                                      'Drug_concentration (µM)': 'dose'})
df_single['source'] = 'single'

# From combination table: drug A
df_combA = drug_comb[['drugA_name', 'drugA Conc (µM)']].copy()
df_combA = df_combA.rename(columns={'drugA_name': 'pert_iname',
                                    'drugA Conc (µM)': 'dose'})
df_combA['source'] = 'combA'

# From combination table: drug B
df_combB = drug_comb[['drugB_name', 'drugB Conc (µM)']].copy()
df_combB = df_combB.rename(columns={'drugB_name': 'pert_iname',
                                    'drugB Conc (µM)': 'dose'})
df_combB['source'] = 'combB'

# Combine all
df_dose_regimes = pd.concat([df_single, df_combA, df_combB], ignore_index=True)

# Deduplicate: keep unique (drug, dose, source)
df_dose_regimes = df_dose_regimes.drop_duplicates()

# Sort for readability
df_dose_regimes = df_dose_regimes.sort_values(['pert_iname', 'dose', 'source']).reset_index(drop=True)


df_all = pd.concat([df_single, df_combA, df_combB], ignore_index=True)

# Drop rows with missing mapping or dose
unmapped = df_all[df_all['pert_iname'].isna()]
missing_dose = df_all[df_all['dose'].isna()]
# (optional) print or log these to inspect typos or missing numbers
# print("Unmapped names:\n", unmapped[['pert_iname_raw','source']].drop_duplicates())
# print("Missing/invalid doses:\n", missing_dose[['pert_iname_raw','dose','source']].drop_duplicates())

df_all = df_all.dropna(subset=['pert_iname','dose']).copy()

# (optional) control floating noise before dedup; adjust decimals if needed
# df_all['dose'] = df_all['dose'].round(6)

# --- 4) Deduplicate by (pert_iname, dose) ONLY, but keep aggregated sources ---
df_dose_regimes_deduplicate = (
    df_all
      .groupby(['pert_iname','dose'], as_index=False)
      .agg(sources=('source', lambda s: ','.join(sorted(set(s)))))
      .sort_values(['pert_iname','dose'])
      .reset_index(drop=True)
)
'''
quick sanity check
'''
df_dose_regimes_deduplicate['pert_iname'].nunique() # 22

'''
screening data builder
'''
def build_adata_screening_from_dose_regimes(
    adata_ht29: "AnnData",
    df_dose_regimes_deduplicate: pd.DataFrame,
    seed: int = 6666,
    time_atol: float = 1e-6,
    verbose: bool = True,
) -> AnnData:
    """
    Build screening AnnData for HT29 based on provided (drug, dose) regimes.
    Each row is assigned a random per-plate mean baseline profile from 24h controls.

    Parameters
    ----------
    adata_ht29 : AnnData
        Source data with obs including:
            - 'pert_time' (numeric/coercible)
            - 'control' (1 = control)
            - 'plate'
            - optional 'cell_id'
    df_dose_regimes_deduplicate : DataFrame
        Must contain columns:
            - 'pert_iname'
            - 'dose'
            - 'sources' (kept for metadata, not used in baseline assignment)
    seed : int
        RNG seed for reproducibility.
    time_atol : float
        Tolerance for matching pert_time.

    Returns
    -------
    AnnData
        New AnnData with baseline expression for each (drug,dose) pair.
        obs includes:
            ['cell_id','pert_iname','dose','pert_time',
             'selected_plate','selected_plate_n_ctrls','sources']
    """

    # ---- Checks ----
    if "pert_time" not in adata_ht29.obs.columns:
        raise KeyError("adata_ht29.obs must contain 'pert_time'")
    if "control" not in adata_ht29.obs.columns:
        raise KeyError("adata_ht29.obs must contain 'control'")
    if "plate" not in adata_ht29.obs.columns:
        raise KeyError("adata_ht29.obs must contain 'plate'")
    for col in ["pert_iname","dose","sources"]:
        if col not in df_dose_regimes_deduplicate.columns:
            raise KeyError(f"df_dose_regimes_deduplicate must include column '{col}'")

    rng = np.random.default_rng(seed)

    # ---- Build control pool at 24h ----
    obs = adata_ht29.obs.copy()
    n_obs = obs.shape[0]
    time_vals = pd.to_numeric(obs["pert_time"], errors="coerce").astype(float)
    is_ctrl = obs["control"].astype(int).eq(1)

    # restrict to ~24h controls
    t_ref = 24.0
    m = is_ctrl & np.isfinite(time_vals) & np.isclose(time_vals, t_ref, rtol=0.0, atol=time_atol)
    if not m.any():
        raise ValueError(f"No controls found at {t_ref}h in adata_ht29.")

    pos_series = pd.Series(np.arange(n_obs), index=obs.index)
    df_sub = obs.loc[m, ["plate"]].copy()
    df_sub["pos"] = pos_series.loc[df_sub.index].values

    plates, means, counts = [], [], []
    X = adata_ht29.X
    for plate, grp in df_sub.groupby("plate", dropna=False, observed=True):
        idxs = grp["pos"].to_numpy()
        if idxs.size == 0:
            continue
        if sparse.issparse(X):
            vec = X[idxs, :].mean(axis=0)
            vec = np.asarray(vec).ravel().astype(np.float32)
        else:
            vec = np.asarray(X[idxs, :], dtype=np.float32).mean(axis=0, dtype=np.float32)
        plates.append(str(plate))
        means.append(vec)
        counts.append(int(idxs.size))

    if len(plates) == 0:
        raise ValueError("No per-plate baselines could be formed at 24h.")

    means = np.vstack(means).astype(np.float32)
    control_pool = {
        "plates": np.array(plates, dtype=object),
        "means": means,
        "counts": np.array(counts, dtype=int),
    }

    # ---- Build new obs and assign random baseline per row ----
    rows = []
    X_rows = []
    for _, row in df_dose_regimes_deduplicate.iterrows():
        n_plate = control_pool["plates"].shape[0]
        pick = int(rng.integers(low=0, high=n_plate))

        rows.append({
            "cell_id": "HT29",
            "pert_iname": str(row["pert_iname"]),
            "dose": float(row["dose"]),
            "pert_time": 96.0,  # screening design
            "selected_plate": control_pool["plates"][pick],
            "selected_plate_n_ctrls": int(control_pool["counts"][pick]),
            "sources": row["sources"],
        })
        X_rows.append(control_pool["means"][pick])

    obs_new = pd.DataFrame(rows)
    X_baseline = np.vstack(X_rows).astype(np.float32)

    obs_names = (
        obs_new["pert_iname"].astype(str)
        + "|dose=" + obs_new["dose"].astype(str)
        + "|t=96"
        + "|plate=" + obs_new["selected_plate"].astype(str)
    )
    obs_new.index = pd.Index(obs_names, name="obs_name")

    var_new = adata_ht29.var.copy()
    adata_screen = AnnData(X=X_baseline, obs=obs_new, var=var_new)

    if verbose:
        n_input = df_dose_regimes_deduplicate.shape[0]
        n_output = adata_screen.n_obs
        n_drugs = obs_new["pert_iname"].nunique()
        n_regimes = obs_new[["pert_iname", "dose"]].drop_duplicates().shape[0]

        print("✅ Screening AnnData built successfully")
        print(f"  Input regimes:       {n_input}")
        print(f"  Output rows:         {n_output}")
        print(f"  Unique drugs:        {n_drugs}")
        print(f"  Unique (drug,dose):  {n_regimes}")
    return adata_screen

adata_new_ht29 = build_adata_screening_from_dose_regimes(
    adata_ht29=adata_ht29_24h,
    df_dose_regimes_deduplicate=df_dose_regimes_deduplicate,  # only 'pert_iname'
    seed=6666
)

adata_new_ht29.obs.columns


adata_new_ht29.write("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/ht29_drugScreen_meanPlate.h5ad")


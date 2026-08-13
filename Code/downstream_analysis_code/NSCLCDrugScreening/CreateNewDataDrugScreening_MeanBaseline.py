import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

adata = sc.read('./Data/FinalData/adataAfterClean.h5ad')
sc.pp.normalize_total(adata)
print(adata.obs['pert_iname'].nunique()) #  17,203 drugs

adata_a549 = adata[adata.obs['cell_id']=='A549']
print(adata_a549.obs['pert_iname'].nunique()) # 11,785 drugs
counts_sorted = (
    adata_a549.obs
    .groupby(["pert_dose", "pert_time"])
    .size()
    .reset_index(name="count")
    .sort_values(by="count", ascending=False)  # largest to smallest
)


adata_a549.write("./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/a549.h5ad")
# adata_a549 = sc.read('./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/a549.h5ad')


def build_adata_new_random_controls_minimal(
    adata_a549: "AnnData",
    drug_table: pd.DataFrame,  # must contain only 'pert_iname' (or at least this column)
    conditions=((10.0, 6.0), (10.0, 24.0), (5.0, 24.0)),
    seed: int = 6666,
    time_atol: float = 1e-6,   # tolerance for matching pert_time
) -> AnnData:
    """
    Create inference-only AnnData for A549. For each (drug, dose, time),
    build a pool of A549 controls at that timepoint, averaged per 'plate';
    then randomly pick one plate-average as the baseline.

    Obs will include:
      ['cell_id','pert_iname','dose','pert_time','selected_plate','selected_plate_n_ctrls']

    Requires in adata_a549.obs:
      - 'pert_time' (numeric/coercible)
      - 'control' (1 = control)
      - 'plate'   (string/categorical)
      - optional 'cell_id' (if missing, assumed all A549)
    drug_table must include: 'pert_iname'
    """

    if "pert_iname" not in drug_table.columns:
        raise KeyError("drug_table must include a 'pert_iname' column.")
    if "control" not in adata_a549.obs.columns:
        raise KeyError("adata_a549.obs must contain a 'control' column (1 = control).")
    if "plate" not in adata_a549.obs.columns:
        raise KeyError("adata_a549.obs must contain a 'plate' column.")
    if "pert_time" not in adata_a549.obs.columns:
        raise KeyError("adata_a549.obs must contain a 'pert_time' column.")

    rng = np.random.default_rng(seed)

    # ---- Build masks and helpers
    obs = adata_a549.obs.copy()
    n_obs = obs.shape[0]
    # If 'cell_id' missing, assume all A549
    if "cell_id" in obs.columns:
        is_a549 = obs["cell_id"].astype(str).eq("A549")
    else:
        is_a549 = pd.Series(True, index=obs.index)

    is_ctrl = obs["control"].astype(int).eq(1)
    time_vals = pd.to_numeric(obs["pert_time"], errors="coerce").astype(float)

    # Map obs index -> integer row position in X
    pos_series = pd.Series(np.arange(n_obs), index=obs.index)

    X = adata_a549.X
    n_genes = adata_a549.shape[1]

    # ---- Precompute pools: for each timepoint, get per-plate averaged baselines
    # control_pools[time] = {"plates": np.array(str), "means": (n_plate, n_gene), "counts": (n_plate,)}
    control_pools = {}
    for _, t in conditions:
        t = float(t)
        m = is_a549 & is_ctrl & np.isfinite(time_vals) & np.isclose(time_vals, t, rtol=0.0, atol=time_atol)
        if not m.any():
            raise ValueError(f"No A549 controls found at time={t}h. Cannot build adata_new.")

        df_sub = obs.loc[m, ["plate"]].copy()
        df_sub["pos"] = pos_series.loc[df_sub.index].values

        plates, means, counts = [], [], []

        # group by plate and average X rows within each plate
        for plate, grp in df_sub.groupby("plate", dropna=False, observed=True):
            idxs = grp["pos"].to_numpy()
            if idxs.size == 0:
                continue

            if sparse.issparse(X):
                # scipy.sparse mean returns 1 x n; convert to 1D float32
                vec = X[idxs, :].mean(axis=0)
                vec = np.asarray(vec).ravel().astype(np.float32)
            else:
                # dense
                vec = np.asarray(X[idxs, :], dtype=np.float32).mean(axis=0, dtype=np.float32)

            plates.append(str(plate))
            means.append(vec)
            counts.append(int(idxs.size))

        if len(plates) == 0:
            raise ValueError(f"No per-plate baselines could be formed at time={t}h.")

        means = np.vstack(means).astype(np.float32)  # (n_plates, n_genes)
        control_pools[t] = {
            "plates": np.array(plates, dtype=object),
            "means": means,
            "counts": np.array(counts, dtype=int),
        }

    # ---- Build screening obs and assemble X by sampling plate-averaged baselines
    rows = []
    X_rows = []

    for _, drug_row in drug_table.iterrows():
        name = str(drug_row["pert_iname"])
        for dose, t in conditions:
            t = float(t)
            pool = control_pools[t]
            n_plate = pool["plates"].shape[0]
            pick = int(rng.integers(low=0, high=n_plate))

            rows.append({
                "cell_id": "A549",
                "pert_iname": name,
                "dose": float(dose),
                "pert_time": t,
                "selected_plate": pool["plates"][pick],
                "selected_plate_n_ctrls": int(pool["counts"][pick]),
            })
            X_rows.append(pool["means"][pick])

    obs_new = pd.DataFrame(rows)

    # Stack selected plate-averaged profiles into X_baseline
    X_baseline = np.vstack(X_rows).astype(np.float32)  # (n_rows, n_genes)

    # Unique obs_names (include plate for traceability)
    obs_names = (
        obs_new["pert_iname"].astype(str)
        + "|" + obs_new["dose"].astype(str)
        + "|" + obs_new["pert_time"].astype(str)
        + "|plate=" + obs_new["selected_plate"].astype(str)
    )
    obs_new.index = pd.Index(obs_names, name="obs_name")

    var_new = adata_a549.var.copy()
    return AnnData(X=X_baseline, obs=obs_new, var=var_new)


# drug_table must have 'pert_iname'
drug_feature = pd.read_csv("./Data/FinalData/compounds_512MFP_wholeDat_fixed.csv")
drug_table = drug_feature[['pert_iname']].copy()

adata_new = build_adata_new_random_controls_minimal(
    adata_a549=adata_a549,
    drug_table=drug_table,  # only 'pert_iname'
    conditions=((10.0, 6.0), (10.0, 24.0), (5.0, 24.0)),
    seed=6666,
)

print(adata_new.shape)
print(adata_new.obs.head())

adata_new.obs
adata_new.obs_names

adata_new.write("./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/a549_drugScreen_meanPlate.h5ad")


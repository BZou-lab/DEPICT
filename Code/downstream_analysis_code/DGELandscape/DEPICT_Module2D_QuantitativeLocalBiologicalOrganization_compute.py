#!/usr/bin/env python3

import os

os.environ.setdefault("OMP_NUM_THREADS","1")

os.environ.setdefault("MKL_NUM_THREADS","1")

os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

os.environ.setdefault("NUMEXPR_NUM_THREADS","1")


# 0. Configuration
from pathlib import Path

MAIN_DIR = Path(
    "~/DEPICT/Code/downstream_analysis_code/DGElandscape"
)
ANALYSIS_DIR = MAIN_DIR / "analysis"
MODULE2C_DIR = ANALYSIS_DIR / "Module2C_PooledMonteCarloTestPredictions_GlobalPredictedDGE"
OUT_ROOT = ANALYSIS_DIR / "Module2D_QuantitativeLocalBiologicalOrganization"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

COORD_PATH = MODULE2C_DIR / "pooled_global_predictedDGE_umap_coordinates.parquet"
EMBEDDING_DIR = MAIN_DIR / "analysis" / "resp_latents_random_split"

RANDOM_SPLITS = [f"random_split{i}" for i in range(1, 6)]
PARTITION = "test"

PCA_VARIANCE_TARGET = 0.80
DISTANCE_METRIC = "cosine"
K_VALUES = [10, 30, 50]
PRIMARY_K = 30

B = 1000
RANDOM_SEED = 20260720
PRIMARY_DURATIONS = {6.0, 24.0}

print("B =", B)
print("Output root:", OUT_ROOT)

# Parallel computation
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
PARALLEL_BACKEND = "loky"



# 1. Imports
import gc
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
})

from joblib import Parallel, delayed



# 2. Load exact Module 2C final population and reconstruct predicted DGE
if not COORD_PATH.exists():
    raise FileNotFoundError(f"Run Module 2C first. Missing: {COORD_PATH}")

final_meta = pd.read_parquet(COORD_PATH).reset_index(drop=True)

required = [
    "source_fold", "row_within_fold", "underlying_profile_id",
    "cell_id", "pert_iname", "pert_iname_norm",
    "broad_moa", "pert_time", "dose",
]
missing = [c for c in required if c not in final_meta.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

parts = []

for split_name in RANDOM_SPLITS:
    fold_mask = final_meta["source_fold"].eq(split_name).to_numpy()
    fold_sub = final_meta.loc[fold_mask].copy()
    if fold_sub.empty:
        continue

    prefix = f"{split_name}_{PARTITION}"
    baseline_path = EMBEDDING_DIR / f"{prefix}_matched_baseline_expression_978_float32.npy"
    pred_path = EMBEDDING_DIR / f"{prefix}_predicted_perturbed_expression_logits_978_float32.npy"

    for p in [baseline_path, pred_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    baseline = np.load(baseline_path, mmap_mode="r")
    pred = np.load(pred_path, mmap_mode="r")

    rows = fold_sub["row_within_fold"].to_numpy(dtype=np.int64)

    if rows.min() < 0 or rows.max() >= baseline.shape[0]:
        raise IndexError(f"{split_name}: row_within_fold out of bounds.")

    fold_dge = np.asarray(pred[rows] - baseline[rows], dtype=np.float32)
    parts.append((fold_sub.index.to_numpy(dtype=np.int64), fold_dge))

all_idx = np.concatenate([x[0] for x in parts])
all_dge = np.concatenate([x[1] for x in parts], axis=0)

if not np.array_equal(np.sort(all_idx), np.arange(len(final_meta))):
    raise RuntimeError("Could not reconstruct each final Module 2C row exactly once.")

pred_dge = all_dge[np.argsort(all_idx)]

print("Final rows:", len(final_meta))
print("Predicted DGE shape:", pred_dge.shape)
print("Unique underlying profiles:", final_meta["underlying_profile_id"].nunique())

del parts, all_idx, all_dge
gc.collect()



# 3. Adaptive PCA >=80% variance
max_components = min(pred_dge.shape[0] - 1, pred_dge.shape[1])

pca = PCA(
    n_components=max_components,
    svd_solver="full",
)
Xp_full = pca.fit_transform(pred_dge)

cumvar = np.cumsum(pca.explained_variance_ratio_)
n_pcs = int(np.searchsorted(cumvar, PCA_VARIANCE_TARGET) + 1)
achieved_var = float(cumvar[n_pcs - 1])

Xp = np.asarray(Xp_full[:, :n_pcs], dtype=np.float32)

pd.DataFrame({
    "pc": np.arange(1, len(cumvar) + 1),
    "individual_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_variance_ratio": cumvar,
}).to_csv(
    OUT_ROOT / "adaptive_PCA_variance_diagnostics.csv",
    index=False,
)

print(f"Selected {n_pcs} PCs; variance explained = {achieved_var:.4f}")

del Xp_full, pred_dge
gc.collect()



# 4. Duplicate-excluded nearest neighbors in PCA space
MAX_K = max(K_VALUES)
RAW_N = min(len(final_meta), max(200, MAX_K + 50))

profile_ids = final_meta["underlying_profile_id"].astype(str).to_numpy()

nn = NearestNeighbors(
    n_neighbors=RAW_N,
    metric=DISTANCE_METRIC,
    n_jobs=-1,
)
nn.fit(Xp)

raw_idx = nn.kneighbors(return_distance=False)

eligible_neighbors = np.full(
    (len(final_meta), MAX_K),
    -1,
    dtype=np.int64,
)

same_profile_excluded = np.zeros(len(final_meta), dtype=np.int32)
need_fallback = []

for i in range(len(final_meta)):
    kept = []
    for j in raw_idx[i]:
        if j == i:
            continue
        if profile_ids[j] == profile_ids[i]:
            same_profile_excluded[i] += 1
            continue
        kept.append(j)
        if len(kept) == MAX_K:
            break

    if len(kept) == MAX_K:
        eligible_neighbors[i] = kept
    else:
        need_fallback.append(i)

if need_fallback:
    print("Fallback full search for", len(need_fallback), "queries")
    full_nn = NearestNeighbors(
        n_neighbors=len(final_meta),
        metric=DISTANCE_METRIC,
        n_jobs=-1,
    )
    full_nn.fit(Xp)
    expanded = full_nn.kneighbors(
        Xp[need_fallback],
        return_distance=False,
    )

    for pos, i in enumerate(need_fallback):
        kept = []
        for j in expanded[pos]:
            if j == i:
                continue
            if profile_ids[j] == profile_ids[i]:
                continue
            kept.append(j)
            if len(kept) == MAX_K:
                break
        if len(kept) < MAX_K:
            raise RuntimeError(f"Query {i} has <{MAX_K} eligible neighbors.")
        eligible_neighbors[i] = kept

audit = pd.DataFrame({
    "underlying_profile_id": profile_ids,
    "same_underlying_profile_candidates_excluded": same_profile_excluded,
})
audit.to_parquet(
    OUT_ROOT / "duplicate_exclusion_neighbor_audit.parquet",
    index=False,
)

print("Total same-underlying-profile candidate exclusions:",
      int(same_profile_excluded.sum()))
print("All queries have", MAX_K, "eligible neighbors.")



# 5. Optimized conditional-permutation utilities
#
# IMPORTANT LOGICAL FIX:
# A stratum is considered "permutable" only if it has:
#   (1) at least 2 rows, AND
#   (2) at least 2 distinct values of the TARGET label.
#
# The previous implementation only checked stratum size. A large stratum containing
# a single target label cannot actually randomize that label, so it must fall back
# to a broader stratum. This is required by our "where feasible" null-model plan.

analysis_meta = final_meta.copy()
analysis_meta["_duration_numeric"] = pd.to_numeric(
    analysis_meta["pert_time"],
    errors="coerce",
)

def build_hierarchical_permutation_plan(labels, df, hierarchy):
    """
    Precompute disjoint index groups for hierarchical conditional permutation.

    Each row is assigned to the most specific stratum that:
      - contains >=2 rows, and
      - contains >=2 distinct target-label values.

    Rows that cannot be permuted at a specific level fall back to the next level.

    Returns
    -------
    groups : list[np.ndarray]
        Disjoint row-index arrays. Labels are independently shuffled within each.
    assigned_level : np.ndarray
        Fallback level used for each row.
    """
    labels = np.asarray(labels)
    remaining = np.ones(len(df), dtype=bool)
    assigned_level = np.full(len(df), -1, dtype=np.int16)
    groups = []

    for level, cols in enumerate(hierarchy):
        if not remaining.any():
            break

        rem_idx = np.flatnonzero(remaining)

        if cols:
            work = df.iloc[rem_idx][cols].copy()
            work["_global_idx"] = rem_idx

            for _, g in work.groupby(cols, dropna=False, sort=False):
                gidx = g["_global_idx"].to_numpy(dtype=np.int64)

                if len(gidx) < 2:
                    continue
                if np.unique(labels[gidx]).size < 2:
                    continue

                groups.append(gidx)
                remaining[gidx] = False
                assigned_level[gidx] = level
        else:
            # Global fallback. It should contain at least two target labels.
            gidx = rem_idx
            if len(gidx) >= 2 and np.unique(labels[gidx]).size >= 2:
                groups.append(gidx)
                remaining[gidx] = False
                assigned_level[gidx] = level

    if remaining.any():
        # This should only happen if the endpoint itself has one unique label.
        bad = int(remaining.sum())
        raise RuntimeError(
            f"{bad} rows could not be assigned to any valid permutation stratum. "
            "Check endpoint label variation and hierarchy."
        )

    # Safety: every row must occur exactly once across groups.
    concat = np.concatenate(groups)
    if len(concat) != len(df) or np.unique(concat).size != len(df):
        raise RuntimeError("Permutation-plan groups are not a complete disjoint partition.")

    return groups, assigned_level


def permute_from_plan(labels, groups, rng):
    """Shuffle labels independently inside precomputed disjoint groups."""
    labels = np.asarray(labels)
    out = labels.copy()
    for idx in groups:
        out[idx] = labels[idx][rng.permutation(len(idx))]
    return out


def agreement_stats_all_k(labels, neighbors_all, k_values):
    """
    Compute mean same-label kNN agreement for all requested k values in one pass.
    """
    labels = np.asarray(labels)

    # Boolean matrix: n_queries x max(k). ~4.6 MB for 92k x 50.
    same = labels[neighbors_all] == labels[:, None]

    # Cumulative number of same-label neighbors lets us get all k efficiently.
    csum = np.cumsum(same, axis=1, dtype=np.int16)

    stats = {}
    per_query = {}
    for k in k_values:
        vals = csum[:, k - 1].astype(np.float64) / float(k)
        per_query[k] = vals
        stats[k] = float(vals.mean())

    return per_query, stats


def inclusive_p_upper(observed, null):
    null = np.asarray(null, dtype=float)
    return (1.0 + np.sum(null >= observed)) / (len(null) + 1.0)


def inclusive_p_lower(observed, null):
    null = np.asarray(null, dtype=float)
    return (1.0 + np.sum(null <= observed)) / (len(null) + 1.0)


def vectorized_drug_cluster_bootstrap(per_query, query_drugs, B, seed):
    """
    Fast perturbagen-cluster bootstrap.

    This is mathematically equivalent to sampling unique drugs with replacement,
    carrying all query rows for each sampled drug, and averaging the concatenated
    profile-level metric. It avoids repeatedly constructing boolean masks/lists.
    """
    rng = np.random.default_rng(seed)

    query_drugs = np.asarray(query_drugs).astype(str)
    unique_drugs, codes = np.unique(query_drugs, return_inverse=True)
    n_drugs = len(unique_drugs)

    cluster_sum = np.bincount(
        codes,
        weights=np.asarray(per_query, dtype=np.float64),
        minlength=n_drugs,
    )
    cluster_n = np.bincount(
        codes,
        minlength=n_drugs,
    ).astype(np.float64)

    # B x n_drugs sampled cluster IDs. For ~1-2k drugs and B=1000 this is small.
    sampled = rng.integers(
        0,
        n_drugs,
        size=(B, n_drugs),
        dtype=np.int32,
    )

    numer = cluster_sum[sampled].sum(axis=1)
    denom = cluster_n[sampled].sum(axis=1)

    return numer / denom


# Precompute and audit permutation plans once.
rng_audit = np.random.default_rng(RANDOM_SEED)

moa_labels_for_plan = analysis_meta["broad_moa"].astype(str).to_numpy()
moa_plan, moa_levels = build_hierarchical_permutation_plan(
    moa_labels_for_plan,
    analysis_meta,
    hierarchy=[
        ["cell_id", "_duration_numeric"],
        ["cell_id"],
        [],
    ],
)

cell_labels_for_plan = analysis_meta["cell_id"].astype(str).to_numpy()
cell_plan, cell_levels = build_hierarchical_permutation_plan(
    cell_labels_for_plan,
    analysis_meta,
    hierarchy=[
        ["broad_moa", "_duration_numeric"],
        ["broad_moa"],
        [],
    ],
)

duration_mask = analysis_meta["_duration_numeric"].isin(PRIMARY_DURATIONS).to_numpy()
duration_meta = analysis_meta.loc[duration_mask].reset_index(drop=True)

duration_labels_for_plan = duration_meta["_duration_numeric"].to_numpy()
duration_plan, duration_levels = build_hierarchical_permutation_plan(
    duration_labels_for_plan,
    duration_meta,
    hierarchy=[
        ["cell_id", "broad_moa"],
        ["cell_id"],
        ["broad_moa"],
        [],
    ],
)

fallback_summary = pd.concat([
    pd.DataFrame({"endpoint": "MoA", "fallback_level": moa_levels}),
    pd.DataFrame({"endpoint": "Cell", "fallback_level": cell_levels}),
    pd.DataFrame({"endpoint": "Duration_6h24h", "fallback_level": duration_levels}),
]).groupby(["endpoint", "fallback_level"], as_index=False).size()

fallback_summary.to_csv(
    OUT_ROOT / "conditional_permutation_fallback_summary.csv",
    index=False,
)

print("Permutation-plan audit:")
print(fallback_summary.to_string(index=False))
print("\nNumber of permutation groups:")
print("  MoA:", len(moa_plan))
print("  Cell:", len(cell_plan))
print("  Duration:", len(duration_plan))



# 6. Optimized + 4-CPU parallel categorical endpoint analysis
#
# Performance improvements:
# 1) Build permutation strata ONCE, not inside every replicate.
# 2) Each B=1000 permutation replicate computes k=10/30/50 simultaneously.
#    The previous code repeated a separate B=1000 permutation loop for each k,
#    causing ~3x unnecessary work.
# 3) Parallelize the B permutation replicates across 4 CPUs.
# 4) Vectorize the perturbagen-cluster bootstrap.

def _one_permutation_replicate(
    replicate_seed,
    labels,
    permutation_plan,
    neighbors_all,
    k_values,
):
    rng = np.random.default_rng(replicate_seed)
    perm = permute_from_plan(
        labels,
        permutation_plan,
        rng,
    )
    _, stats = agreement_stats_all_k(
        perm,
        neighbors_all,
        k_values,
    )
    return np.asarray(
        [stats[k] for k in k_values],
        dtype=np.float64,
    )


def run_categorical_endpoint_parallel(
    endpoint,
    labels,
    meta_df,
    neighbors_all,
    permutation_plan,
    B,
    seed,
    n_jobs=4,
):
    labels = np.asarray(labels)

    # Observed values for all k in ONE pass.
    per_query_by_k, observed_by_k = agreement_stats_all_k(
        labels,
        neighbors_all,
        K_VALUES,
    )

    # Independent deterministic seeds for every replicate.
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(B)
    replicate_seeds = [
        int(s.generate_state(1, dtype=np.uint64)[0])
        for s in child_seeds
    ]

    print(
        f"\n{endpoint}: running B={B} conditional permutations "
        f"for k={K_VALUES} simultaneously using {n_jobs} CPUs..."
    )

    # Shape after stack: B x len(K_VALUES)
    null_matrix = np.vstack(
        Parallel(
            n_jobs=n_jobs,
            backend=PARALLEL_BACKEND,
            verbose=5,
            batch_size="auto",
            max_nbytes="10M",
        )(
            delayed(_one_permutation_replicate)(
                rs,
                labels,
                permutation_plan,
                neighbors_all,
                K_VALUES,
            )
            for rs in replicate_seeds
        )
    )

    qdrugs = meta_df["pert_iname_norm"].astype(str).to_numpy()
    n_unique_drugs = int(np.unique(qdrugs).size)

    rows = []
    nulls = {}

    for col, k in enumerate(K_VALUES):
        observed = observed_by_k[k]
        null_stats = null_matrix[:, col]
        null_mean = float(null_stats.mean())

        per_query = per_query_by_k[k]

        boot = vectorized_drug_cluster_bootstrap(
            per_query=per_query,
            query_drugs=qdrugs,
            B=B,
            seed=seed + 10000 + k,
        )
        ci_lo, ci_hi = np.quantile(
            boot,
            [0.025, 0.975],
        )

        ratio = (
            observed / null_mean
            if null_mean > 0
            else np.nan
        )

        rows.append({
            "endpoint": endpoint,
            "k": k,
            "n_queries": len(meta_df),
            "n_unique_query_drugs": n_unique_drugs,
            "observed_agreement": observed,
            "null_mean_agreement": null_mean,
            "absolute_observed_minus_null": observed - null_mean,
            "enrichment_ratio_observed_over_null": ratio,
            "bootstrap_ci95_low_observed_agreement": float(ci_lo),
            "bootstrap_ci95_high_observed_agreement": float(ci_hi),
            "empirical_p_upper": inclusive_p_upper(
                observed,
                null_stats,
            ),
            "B_permutation": B,
            "B_bootstrap": B,
        })

        nulls[k] = null_stats.copy()

    return pd.DataFrame(rows), nulls


# Run MoA and cell endpoints.
# They are run sequentially at the endpoint level, while each endpoint uses 4 CPUs.
# This avoids nested parallelism and excessive RAM duplication.

moa_results, moa_nulls = run_categorical_endpoint_parallel(
    endpoint="Broad MoA",
    labels=analysis_meta["broad_moa"].astype(str).to_numpy(),
    meta_df=analysis_meta,
    neighbors_all=eligible_neighbors,
    permutation_plan=moa_plan,
    B=B,
    seed=RANDOM_SEED + 101,
    n_jobs=N_JOBS,
)

print("\nMoA results:")
print(moa_results.to_string(index=False))

cell_results, cell_nulls = run_categorical_endpoint_parallel(
    endpoint="Cell line",
    labels=analysis_meta["cell_id"].astype(str).to_numpy(),
    meta_df=analysis_meta,
    neighbors_all=eligible_neighbors,
    permutation_plan=cell_plan,
    B=B,
    seed=RANDOM_SEED + 202,
    n_jobs=N_JOBS,
)

print("\nCell results:")
print(cell_results.to_string(index=False))



# 7. Duration endpoint: 6 h vs 24 h only, with duration-subset neighbors
duration_idx = np.flatnonzero(duration_mask)
Xp_duration = Xp[duration_idx]
duration_profile_ids = profile_ids[duration_idx]

RAW_DUR = min(len(duration_idx), max(200, MAX_K + 50))

nn_dur = NearestNeighbors(
    n_neighbors=RAW_DUR,
    metric=DISTANCE_METRIC,
    n_jobs=-1,
)
nn_dur.fit(Xp_duration)
raw_dur_idx = nn_dur.kneighbors(return_distance=False)

duration_neighbors = np.full(
    (len(duration_idx), MAX_K),
    -1,
    dtype=np.int64,
)

for i in range(len(duration_idx)):
    kept = []
    for j in raw_dur_idx[i]:
        if j == i:
            continue
        if duration_profile_ids[j] == duration_profile_ids[i]:
            continue
        kept.append(j)
        if len(kept) == MAX_K:
            break

    if len(kept) < MAX_K:
        raise RuntimeError(
            f"Duration query {i} has <{MAX_K} eligible neighbors."
        )
    duration_neighbors[i] = kept


duration_results, duration_nulls = run_categorical_endpoint_parallel(
    endpoint="Duration (6h vs 24h)",
    labels=duration_meta["_duration_numeric"].to_numpy(),
    meta_df=duration_meta,
    neighbors_all=duration_neighbors,
    permutation_plan=duration_plan,
    B=B,
    seed=RANDOM_SEED + 303,
    n_jobs=N_JOBS,
)

categorical_results = pd.concat(
    [moa_results, cell_results, duration_results],
    ignore_index=True,
)

categorical_results.to_csv(
    OUT_ROOT / "categorical_local_organization_results.csv",
    index=False,
)

print("\nDuration results:")
print(duration_results.to_string(index=False))



# 8. Dose endpoint: same-drug neighborhood organization — FIXED + OPTIMIZED
#
# Fix for sklearn error:
# When kneighbors() is called with X=None on the training data, sklearn internally
# excludes each sample itself and therefore requires n_neighbors < n_samples_fit.
# The previous code used n_neighbors=len(idx), which triggers:
#   ValueError: Expected n_neighbors < n_samples_fit
#
# We now use n_neighbors=len(idx)-1 and cache each drug's complete within-drug
# neighbor ordering ONCE. The k=10/30/50 analyses reuse this cache.
#
# Statistical design is unchanged:
# - neighbors must be the same drug;
# - repeated predictions with the same underlying_profile_id are excluded;
# - observed statistic = mean absolute log10-dose difference to k nearest same-drug neighbors;
# - null = k random eligible same-drug profiles per query;
# - B=1000;
# - drug-cluster bootstrap for uncertainty.

dose = pd.to_numeric(
    analysis_meta["dose"],
    errors="coerce",
).to_numpy(dtype=float)

valid_dose = np.isfinite(dose) & (dose > 0)

log_dose = np.full(len(dose), np.nan, dtype=float)
log_dose[valid_dose] = np.log10(dose[valid_dose])

drug_labels = analysis_meta["pert_iname_norm"].astype(str).to_numpy()

drug_to_idx = defaultdict(list)
for i, d in enumerate(drug_labels):
    if valid_dose[i]:
        drug_to_idx[d].append(i)

drug_to_idx = {
    d: np.asarray(idx, dtype=np.int64)
    for d, idx in drug_to_idx.items()
}

# -------------------------------------------------------------------------
# Precompute complete same-drug neighbor order once per drug.
# -------------------------------------------------------------------------
same_drug_raw_order = {}

for drug, idx in drug_to_idx.items():
    n = len(idx)

    # Need at least 2 profiles to define a neighbor.
    if n < 2:
        continue

    model = NearestNeighbors(
        n_neighbors=n - 1,   # IMPORTANT: must be strictly < n_samples_fit for X=None
        metric=DISTANCE_METRIC,
        n_jobs=N_JOBS,
    )
    model.fit(Xp[idx])

    # Because X=None, sklearn treats this as training-set neighbors
    # and excludes the query point itself.
    raw_local = model.kneighbors(
        return_distance=False,
    )

    same_drug_raw_order[drug] = (
        idx,
        raw_local,
    )

print(
    "Precomputed same-drug neighbor order for",
    len(same_drug_raw_order),
    "drugs."
)


def same_drug_knn(k):
    """
    Return queries having at least k eligible same-drug neighbors after excluding
    repeated predictions of the same underlying experimental profile.
    """
    qidx = []
    neigh_rows = []

    for drug, (idx, raw_local) in same_drug_raw_order.items():
        if len(idx) < k + 1:
            continue

        for local_i, global_i in enumerate(idx):
            kept = []

            for local_j in raw_local[local_i]:
                global_j = idx[local_j]

                # Self is already excluded by sklearn when X=None,
                # but keep this guard for safety.
                if global_j == global_i:
                    continue

                # Critical: repeated Monte Carlo predictions of the same
                # underlying profile cannot count as biological neighbors.
                if profile_ids[global_j] == profile_ids[global_i]:
                    continue

                kept.append(global_j)

                if len(kept) == k:
                    break

            if len(kept) == k:
                qidx.append(global_i)
                neigh_rows.append(kept)

    if not qidx:
        raise RuntimeError(
            f"No eligible dose queries had {k} duplicate-excluded same-drug neighbors."
        )

    return (
        np.asarray(qidx, dtype=np.int64),
        np.asarray(neigh_rows, dtype=np.int64),
    )


# -------------------------------------------------------------------------
# Precompute eligible random-null pools once for every query.
# This avoids rebuilding boolean masks inside B x n_queries loops.
# -------------------------------------------------------------------------
eligible_same_drug_pool = {}

for drug, idx in drug_to_idx.items():
    for qi in idx:
        pool = idx[
            (idx != qi)
            & (profile_ids[idx] != profile_ids[qi])
        ]
        eligible_same_drug_pool[int(qi)] = pool


def _dose_null_one_replicate(
    seed,
    qidx,
    k,
):
    """
    One matched-null replicate:
    for every query, randomly sample k eligible same-drug profiles without replacement.
    """
    rng = np.random.default_rng(seed)
    total = 0.0

    for qi in qidx:
        eligible = eligible_same_drug_pool[int(qi)]

        if len(eligible) < k:
            raise RuntimeError(
                f"Dose null query {qi} has only {len(eligible)} eligible "
                f"same-drug profiles for k={k}."
            )

        chosen = rng.choice(
            eligible,
            size=k,
            replace=False,
        )

        total += np.mean(
            np.abs(
                log_dose[chosen]
                - log_dose[qi]
            )
        )

    return total / len(qidx)


def run_dose(
    k,
    B,
    seed,
    n_jobs=4,
):
    qidx, neigh = same_drug_knn(k)

    per_query = np.mean(
        np.abs(
            log_dose[neigh]
            - log_dose[qidx, None]
        ),
        axis=1,
    )

    observed = float(
        per_query.mean()
    )

    print(
        f"\nDose k={k}: {len(qidx):,} eligible queries from "
        f"{np.unique(drug_labels[qidx]).size:,} drugs."
    )
    print(
        f"Running B={B} matched same-drug null replicates "
        f"using {n_jobs} CPUs..."
    )

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(B)

    replicate_seeds = [
        int(
            s.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for s in child_seeds
    ]

    null = np.asarray(
        Parallel(
            n_jobs=n_jobs,
            backend=PARALLEL_BACKEND,
            verbose=5,
            batch_size="auto",
        )(
            delayed(_dose_null_one_replicate)(
                rs,
                qidx,
                k,
            )
            for rs in replicate_seeds
        ),
        dtype=float,
    )

    null_mean = float(
        null.mean()
    )

    organization = (
        1.0
        - observed / null_mean
        if null_mean > 0
        else np.nan
    )

    # Drug-cluster bootstrap, vectorized using the optimized helper from Cell 6.
    qdrugs = drug_labels[qidx]

    boot = vectorized_drug_cluster_bootstrap(
        per_query=per_query,
        query_drugs=qdrugs,
        B=B,
        seed=seed + 10000,
    )

    ci_lo, ci_hi = np.quantile(
        boot,
        [0.025, 0.975],
    )

    result = {
        "endpoint": "Dose",
        "k": k,
        "n_queries": len(qidx),
        "n_unique_query_drugs": len(np.unique(qdrugs)),
        "observed_mean_abs_log10dose_difference": observed,
        "null_mean_abs_log10dose_difference": null_mean,
        "absolute_null_minus_observed": null_mean - observed,
        "dose_organization_1_minus_observed_over_null": organization,
        "bootstrap_ci95_low_observed_distance": float(ci_lo),
        "bootstrap_ci95_high_observed_distance": float(ci_hi),
        "empirical_p_lower": inclusive_p_lower(
            observed,
            null,
        ),
        "B_random_null": B,
        "B_bootstrap": B,
    }

    return (
        result,
        null,
    )


dose_rows = []
dose_nulls = {}

for k in K_VALUES:
    res, null = run_dose(
        k=k,
        B=B,
        seed=RANDOM_SEED + 400 + k,
        n_jobs=N_JOBS,
    )

    dose_rows.append(res)
    dose_nulls[k] = null


dose_results = pd.DataFrame(
    dose_rows
)

dose_results.to_csv(
    OUT_ROOT / "dose_local_organization_results.csv",
    index=False,
)

print("\nDose results:")
print(
    dose_results.to_string(
        index=False
    )
)



# 9. Save null distributions
null_frames = []

for endpoint, null_dict in [
    ("Broad MoA", moa_nulls),
    ("Cell line", cell_nulls),
    ("Duration (6h vs 24h)", duration_nulls),
    ("Dose", dose_nulls),
]:
    for k, arr in null_dict.items():
        null_frames.append(pd.DataFrame({
            "endpoint": endpoint,
            "k": k,
            "replicate": np.arange(1, len(arr) + 1),
            "null_statistic": arr,
        }))

all_nulls = pd.concat(null_frames, ignore_index=True)
all_nulls.to_parquet(
    OUT_ROOT / "all_null_distributions_B1000.parquet",
    index=False,
)

print("Saved", len(all_nulls), "null replicates.")



# 10. Primary k=30 summary
primary_cat = categorical_results.loc[
    categorical_results["k"].eq(PRIMARY_K)
].copy()

rows = []

for _, r in primary_cat.iterrows():
    rows.append({
        "endpoint": r["endpoint"],
        "k": PRIMARY_K,
        "effect_definition": "observed/null agreement ratio",
        "normalized_effect": r["enrichment_ratio_observed_over_null"] - 1.0,
        "observed_raw": r["observed_agreement"],
        "null_raw": r["null_mean_agreement"],
        "absolute_difference": r["absolute_observed_minus_null"],
        "empirical_p": r["empirical_p_upper"],
        "n_queries": r["n_queries"],
        "n_unique_query_drugs": r["n_unique_query_drugs"],
    })

d = dose_results.loc[dose_results["k"].eq(PRIMARY_K)].iloc[0]

rows.append({
    "endpoint": "Dose",
    "k": PRIMARY_K,
    "effect_definition": "1 - observed/null log-dose distance",
    "normalized_effect": d["dose_organization_1_minus_observed_over_null"],
    "observed_raw": d["observed_mean_abs_log10dose_difference"],
    "null_raw": d["null_mean_abs_log10dose_difference"],
    "absolute_difference": d["absolute_null_minus_observed"],
    "empirical_p": d["empirical_p_lower"],
    "n_queries": d["n_queries"],
    "n_unique_query_drugs": d["n_unique_query_drugs"],
})

primary_summary = pd.DataFrame(rows)
primary_summary.to_csv(
    OUT_ROOT / "primary_k30_publication_summary.csv",
    index=False,
)

print(primary_summary.to_string(index=False))


(OUT_ROOT / "BATCH_COMPUTATION_COMPLETE.txt").write_text("Module 2D heavy computation completed successfully.\n")
print("BATCH COMPUTATION COMPLETE:", OUT_ROOT, flush=True)

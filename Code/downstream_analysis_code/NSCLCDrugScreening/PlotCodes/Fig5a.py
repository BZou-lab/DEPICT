# %%
"""Figure 2A: rank-overlap validation. Run section-by-section in PyCharm."""

# %%
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SCRIPT_DIR = Path(
    "/Users/meishengxiao/PycharmProjects/PhD_disser/"
    "Experiments/downstreamTaskCode/lung_cancer_A549/DrugRepurposing/"
    "fig2_drug_screening_scripts"
)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

COMMON_FILE = SCRIPT_DIR / "fig2_common_interactive.py"
if not COMMON_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find fig2_common_interactive.py at {COMMON_FILE}. "
        "Please keep fig2_common_interactive.py in the same fig2_drug_screening_scripts folder as this panel script."
    )

from fig2_common_interactive import (
    set_publication_style,
    load_screening_tables,
    restrict_predicted_to_observed_drugs,
    best_per_drug,
    normalize_drug_name,
    SOURCE_COLORS,
    TABLE_DIR,
    save_figure,
)
set_publication_style()

# %%
final_df_pred, final_df_orig = load_screening_tables()
final_df_pred_observed_drugs = restrict_predicted_to_observed_drugs(final_df_pred, final_df_orig)
pred_ranked = best_per_drug(final_df_pred_observed_drugs)
orig_ranked = best_per_drug(final_df_orig)

print("Observed unique drugs:", orig_ranked["pert_iname"].nunique())
print("Predicted unique drugs restricted to observed drugs:", pred_ranked["pert_iname"].nunique())
print(orig_ranked[["rank", "pert_iname", "reverse_score"]].head(20))
print(pred_ranked[["rank", "pert_iname", "reverse_score"]].head(20))

# %%
merged = orig_ranked[["pert_iname", "drug_key", "reverse_score", "rank"]].rename(
    columns={"pert_iname": "observed_drug", "reverse_score": "observed_reverse_score", "rank": "observed_rank"}
).merge(
    pred_ranked[["drug_key", "pert_iname", "reverse_score", "rank"]].rename(
        columns={"pert_iname": "predicted_drug", "reverse_score": "predicted_reverse_score", "rank": "predicted_rank"}
    ),
    on="drug_key",
    how="inner",
)

rho_score, pval_score = spearmanr(merged["observed_reverse_score"], merged["predicted_reverse_score"], nan_policy="omit")
rho_rank, pval_rank = spearmanr(merged["observed_rank"], merged["predicted_rank"], nan_policy="omit")
print(f"Reverse-score Spearman rho = {rho_score:.4f}, p = {pval_score:.3e}")
print(f"Rank Spearman rho = {rho_rank:.4f}, p = {pval_rank:.3e}")
merged.to_csv(TABLE_DIR / "fig2a_shared_drug_rank_score_concordance.csv", index=False)

# %%
# Top-k list overlap at selected checkpoints for reporting in the figure label.
overlap_rows = []
for k in [10, 20, 50, 100, 500, 1000]:
    obs_keys = set(orig_ranked.head(k)["pert_iname"].map(normalize_drug_name))
    pred_keys = set(pred_ranked.head(k)["pert_iname"].map(normalize_drug_name))
    overlap_n = len(obs_keys.intersection(pred_keys))
    overlap_rows.append({"top_k": k, "overlap_n": overlap_n, "overlap_fraction": overlap_n / k})
overlap = pd.DataFrame(overlap_rows)
print(overlap)
overlap.to_csv(TABLE_DIR / "fig2a_topk_overlap_summary.csv", index=False)

# %%
# Smooth recovery curves: for each predicted rank cutoff from 1 to 1000,
# calculate how many observed top-N drugs are recovered in the predicted top-k list.
max_rank_cutoff = 1000
rank_cutoffs = np.arange(1, max_rank_cutoff + 1)
reference_top_ns = [10, 20, 50]

reference_key_sets = {
    n: set(orig_ranked.head(n)["pert_iname"].map(normalize_drug_name))
    for n in reference_top_ns
}

predicted_keys_in_rank_order = pred_ranked.head(max_rank_cutoff)["pert_iname"].map(normalize_drug_name).tolist()

recovery_rows = []
running_predicted_keys = set()
for k, drug_key in enumerate(predicted_keys_in_rank_order, start=1):
    running_predicted_keys.add(drug_key)
    row = {"predicted_rank_cutoff": k}
    for n in reference_top_ns:
        recovered_n = len(reference_key_sets[n].intersection(running_predicted_keys))
        row[f"observed_top{n}_recovered"] = recovered_n
        row[f"observed_top{n}_recovery_fraction"] = recovered_n / n
        row[f"observed_top{n}_random_expectation"] = min(k / len(pred_ranked), 1.0)
    recovery_rows.append(row)

recovery = pd.DataFrame(recovery_rows)
print(recovery.tail())
recovery.to_csv(TABLE_DIR / "fig2a_observed_top10_top20_top50_recovery_by_predicted_rank_1to1000.csv", index=False)

# %%
fig, ax = plt.subplots(figsize=(3.75, 2.75))

curve_specs = [
    (10, "#2166AC", "Observed top-10"),
    (20, SOURCE_COLORS["DEPICT-predicted LINCS"], "Observed top-20"),
    (50, "#1B7837", "Observed top-50"),
]

for n, color, label in curve_specs:
    ax.plot(
        recovery["predicted_rank_cutoff"],
        recovery[f"observed_top{n}_recovery_fraction"],
        lw=1.7,
        color=color,
        label=label,
    )

# Random expectation is the same fractional expectation for top-10, top-20, and top-50
# under random ranking, because E[recovered fraction] = k / total_drugs.
ax.plot(
    recovery["predicted_rank_cutoff"],
    recovery["observed_top20_random_expectation"],
    ls="--",
    lw=1.0,
    color="0.45",
    label="Random expectation",
)

ax.set_xlim(1, max_rank_cutoff)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("Predicted LINCS rank cutoff")
ax.set_ylabel("Fraction of observed top-N recovered")
ax.set_title("A. Recovery of observed top-ranked candidates")
ax.grid(axis="y", color="0.90", lw=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

summary_text = (
    f"Top-10 overlap: {int(overlap.loc[overlap['top_k'] == 10, 'overlap_n'].iloc[0])}/10\n"
    f"Top-20 overlap: {int(overlap.loc[overlap['top_k'] == 20, 'overlap_n'].iloc[0])}/20\n"
    f"Top-50 overlap: {int(overlap.loc[overlap['top_k'] == 50, 'overlap_n'].iloc[0])}/50\n"
    f"Score Spearman ρ={rho_score:.2f}"
)
ax.text(
    0.7,
    0.6,
    summary_text,
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=7,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.85", lw=0.6),
)
ax.legend(frameon=False, loc="lower right")
save_figure(fig, "fig2a_rank_overlap_validation_top10_top20_top50_smooth")

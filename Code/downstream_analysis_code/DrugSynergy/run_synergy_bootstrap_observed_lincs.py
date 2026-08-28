from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from synergy_bootstrap_utils import (
    DEFAULT_MODELS,
    compute_pooled_signatures_from_anndata,
    ensure_dir,
    match_reference_to_signatures,
    run_synergy_analysis,
    standardize_signature_table,
)


def find_project_dir() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "data").is_dir():
            return parent
    return script_path.parent.parent


PROJECT_DIR = Path("~/DEPICT/Code/downstream_analysis_code/DrugSynergy")
DATA_DIR = PROJECT_DIR / "Data"
RESULTS_ROOT = PROJECT_DIR / "results" / "synergy_bootstrap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HT29 drug synergy classification with observed LINCS proxy signatures "
            "and bootstrap uncertainty on leave-one-out prediction rows."
        )
    )
    parser.add_argument(
        "--depict-reference-signatures",
        type=Path,
        default=DATA_DIR / "diff_geneExp_pred_ht29.csv",
        help=(
            "DEPICT signature CSV used only for the reference drug-dose-time grid. "
            "Observed LINCS signatures are matched to this grid."
        ),
    )
    parser.add_argument(
        "--merged-adata",
        type=Path,
        default=DATA_DIR / "HT29_22drugs_wControl.h5ad",
        help="AnnData file containing observed HT29 LINCS treated rows plus matched controls.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DATA_DIR / "HT29_allpairs_LoeweCI_labels.csv",
        help="CSV containing drug-pair synergy labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_ROOT / "observed_lincs",
        help="Directory where result tables will be written.",
    )
    parser.add_argument("--n-components", type=int, default=50, help="Number of PCA components.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap resamples.")
    parser.add_argument("--random-state", type=int, default=6666, help="Random seed.")
    parser.add_argument("--rf-n-estimators", type=int, default=200, help="Random forest trees.")
    parser.add_argument("--rf-n-jobs", type=int, default=-1, help="Random forest parallel jobs.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold.")
    parser.add_argument(
        "--curve-grid-size",
        type=int,
        default=101,
        help="Number of grid points used to save interpolated ROC and PR curves.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=list(DEFAULT_MODELS),
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--stratified-bootstrap",
        action="store_true",
        help="Bootstrap within each class instead of over all pair rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    df_diffexp_pred_read = pd.read_csv(args.depict_reference_signatures, index_col=0)
    merged_adata_read = ad.read_h5ad(args.merged_adata)
    drug_doublet_label = pd.read_csv(args.labels)

    signatures_df, counts_df = compute_pooled_signatures_from_anndata(merged_adata_read)
    reference_meta = df_diffexp_pred_read.loc[:, ["pert_iname", "dose", "pert_time"]].copy()
    matched_df = match_reference_to_signatures(reference_meta, signatures_df)

    signatures_df.to_csv(output_dir / "observed_pooled_signatures.csv", index=False)
    counts_df.to_csv(output_dir / "observed_signature_replicate_counts.csv")
    matched_df.to_csv(output_dir / "observed_matched_to_depict_reference.csv", index=False)

    matched_meta_cols = [
        "pert_iname_ref",
        "dose_ref",
        "pert_time_ref",
        "pert_iname_matched",
        "pert_dose_matched",
        "pert_time_matched",
    ]
    gene_cols = [col for col in matched_df.columns if col not in matched_meta_cols]

    complete_mask = matched_df.loc[:, gene_cols].notna().all(axis=1)
    complete_mask &= matched_df["pert_iname_ref"].notna()
    complete_mask &= matched_df["dose_ref"].notna()
    if not complete_mask.all():
        unmatched = matched_df.loc[~complete_mask].copy()
        unmatched.to_csv(output_dir / "observed_unmatched_reference_rows.csv", index=False)
        print(f"[observed_lincs] Dropping {unmatched.shape[0]} unmatched reference rows before PCA.")

    observed_for_model = matched_df.loc[complete_mask, [*gene_cols, *matched_meta_cols]].copy()
    observed_for_model = observed_for_model.rename(
        columns={
            "pert_iname_ref": "pert_iname",
            "dose_ref": "dose",
            "pert_time_ref": "pert_time",
        }
    )
    observed_for_model = observed_for_model.loc[:, [*gene_cols, "pert_iname", "dose", "pert_time"]]
    observed_for_model = standardize_signature_table(
        observed_for_model,
        gene_cols=gene_cols,
        drug_col="pert_iname",
        dose_col="dose",
        time_col="pert_time",
    )

    run_synergy_analysis(
        source_name="observed_lincs",
        signature_df=observed_for_model,
        label_df=drug_doublet_label,
        gene_cols=gene_cols,
        output_dir=output_dir,
        n_components=args.n_components,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        model_names=args.models,
        rf_n_estimators=args.rf_n_estimators,
        rf_n_jobs=args.rf_n_jobs,
        threshold=args.threshold,
        curve_grid_size=args.curve_grid_size,
        stratified_bootstrap=args.stratified_bootstrap,
    )


if __name__ == "__main__":
    main()

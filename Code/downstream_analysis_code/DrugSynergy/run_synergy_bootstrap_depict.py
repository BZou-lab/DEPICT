from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from synergy_bootstrap_utils import (
    DEFAULT_MODELS,
    get_gene_columns,
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
            "Run HT29 drug synergy classification with DEPICT-predicted signatures "
            "and bootstrap uncertainty on leave-one-out prediction rows."
        )
    )
    parser.add_argument(
        "--depict-signatures",
        type=Path,
        default=DATA_DIR / "diff_geneExp_pred_ht29.csv",
        help="CSV containing DEPICT-predicted drug signatures.",
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
        default=RESULTS_ROOT / "depict",
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

    df_diffexp_pred_read = pd.read_csv(args.depict_signatures, index_col=0)
    drug_doublet_label = pd.read_csv(args.labels)

    gene_cols = get_gene_columns(
        df_diffexp_pred_read,
        metadata_cols=["pert_iname", "dose", "pert_time"],
    )
    signature_df = standardize_signature_table(
        df_diffexp_pred_read,
        gene_cols=gene_cols,
        drug_col="pert_iname",
        dose_col="dose",
        time_col="pert_time",
    )

    run_synergy_analysis(
        source_name="depict",
        signature_df=signature_df,
        label_df=drug_doublet_label,
        gene_cols=gene_cols,
        output_dir=args.output_dir,
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

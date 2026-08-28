from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneOut


METRIC_COLUMNS = ["roc_auc", "pr_auc", "accuracy", "macro_f1"]
DEFAULT_MODELS = ("logistic_regression", "random_forest")
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def clean_drug_name(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(value).strip().lower()


def dose_key(value: object, ndigits: int = 12) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return round(float(numeric), ndigits)


def get_gene_columns(signature_df: pd.DataFrame, metadata_cols: Iterable[str]) -> list[str]:
    metadata = set(metadata_cols)
    gene_cols = [col for col in signature_df.columns if col not in metadata]
    if not gene_cols:
        raise ValueError("No gene columns were found after excluding metadata columns.")
    return gene_cols


def standardize_signature_table(
    signature_df: pd.DataFrame,
    gene_cols: Sequence[str],
    drug_col: str = "pert_iname",
    dose_col: str = "dose",
    time_col: str | None = "pert_time",
) -> pd.DataFrame:
    required = [drug_col, dose_col, *gene_cols]
    if time_col is not None and time_col in signature_df.columns:
        required.append(time_col)

    missing = [col for col in required if col not in signature_df.columns]
    if missing:
        raise KeyError(f"Signature table is missing required columns: {missing}")

    keep_cols = [*gene_cols, drug_col, dose_col]
    if time_col is not None and time_col in signature_df.columns:
        keep_cols.append(time_col)

    out = signature_df.loc[:, keep_cols].copy()
    rename_map = {drug_col: "pert_iname", dose_col: "dose"}
    if time_col is not None and time_col in out.columns:
        rename_map[time_col] = "pert_time"
    out = out.rename(columns=rename_map)

    out["pert_iname"] = out["pert_iname"].astype(str).str.strip()
    out["dose"] = pd.to_numeric(out["dose"], errors="coerce")
    if "pert_time" in out.columns:
        out["pert_time"] = pd.to_numeric(out["pert_time"], errors="coerce")

    if out["dose"].isna().any():
        bad = out.loc[out["dose"].isna(), ["pert_iname", "dose"]].head()
        raise ValueError(f"Some signature rows have non-numeric dose values:\n{bad}")

    numeric_genes = out.loc[:, gene_cols].apply(pd.to_numeric, errors="coerce")
    if numeric_genes.isna().any().any():
        bad_cols = numeric_genes.columns[numeric_genes.isna().any()].tolist()[:10]
        raise ValueError(f"Gene columns contain non-numeric or missing values: {bad_cols}")

    out.loc[:, gene_cols] = numeric_genes
    return out


def fit_pca_profile_features(
    signature_df: pd.DataFrame,
    gene_cols: Sequence[str],
    n_components: int = 50,
    random_state: int = 6666,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_samples = signature_df.shape[0]
    n_genes = len(gene_cols)
    effective_components = min(n_components, n_samples, n_genes)
    if effective_components < n_components:
        print(
            f"Requested {n_components} PCs, but only {effective_components} are possible "
            f"for {n_samples} profiles and {n_genes} genes."
        )

    x_gene = signature_df.loc[:, gene_cols].to_numpy(dtype=float)
    pca = PCA(n_components=effective_components, random_state=random_state)
    x_pca = pca.fit_transform(x_gene)
    pc_cols = [f"PC{i + 1}" for i in range(effective_components)]

    profile_features = pd.DataFrame(x_pca, index=signature_df.index, columns=pc_cols)
    profile_features["pert_iname"] = signature_df["pert_iname"].to_numpy()
    profile_features["dose"] = signature_df["dose"].to_numpy()
    if "pert_time" in signature_df.columns:
        profile_features["pert_time"] = signature_df["pert_time"].to_numpy()

    pca_variance = pd.DataFrame(
        {
            "component": pc_cols,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    pca_variance["cumulative_explained_variance_ratio"] = pca_variance[
        "explained_variance_ratio"
    ].cumsum()

    return profile_features, pca_variance


def _drug_name_map(available_drugs: Sequence[object]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for drug in available_drugs:
        key = clean_drug_name(drug)
        if key is not None and key not in mapping:
            mapping[key] = str(drug).strip()
    return mapping


def prepare_synergy_labels(
    label_df: pd.DataFrame,
    available_drugs: Sequence[object],
    positive_label: str = "synergy",
    negative_label: str = "antagonism",
) -> pd.DataFrame:
    required = ["drug1", "drug2", "drug1_conc", "drug2_conc", "label"]
    missing = [col for col in required if col not in label_df.columns]
    if missing:
        raise KeyError(f"Label table is missing required columns: {missing}")

    labels = label_df.copy().reset_index().rename(columns={"index": "label_index"})
    labels["label_clean"] = labels["label"].astype(str).str.strip().str.lower()
    positive_label = positive_label.lower()
    negative_label = negative_label.lower()

    keep_mask = labels["label_clean"].isin([positive_label, negative_label])
    dropped = labels.loc[~keep_mask, "label"].value_counts(dropna=False)
    if not dropped.empty:
        print(f"Dropping non-binary labels before modeling:\n{dropped}")
    labels = labels.loc[keep_mask].copy()

    labels["y"] = np.where(labels["label_clean"].eq(positive_label), 1, 0)
    labels["drug1_key"] = labels["drug1"].apply(clean_drug_name)
    labels["drug2_key"] = labels["drug2"].apply(clean_drug_name)
    labels["drug1_conc_num"] = pd.to_numeric(labels["drug1_conc"], errors="coerce")
    labels["drug2_conc_num"] = pd.to_numeric(labels["drug2_conc"], errors="coerce")

    name_map = _drug_name_map(available_drugs)
    labels["drug1_matched"] = labels["drug1_key"].map(name_map)
    labels["drug2_matched"] = labels["drug2_key"].map(name_map)

    return labels


def build_pair_feature_table(
    profile_features: pd.DataFrame,
    label_df: pd.DataFrame,
    pc_cols: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    if pc_cols is None:
        pc_cols = [col for col in profile_features.columns if col.startswith("PC")]
    if not pc_cols:
        raise ValueError("No PCA columns found in profile feature table.")

    labels = prepare_synergy_labels(label_df, profile_features["pert_iname"].unique())

    profiles = profile_features.copy()
    profiles["_drug_key"] = profiles["pert_iname"].apply(clean_drug_name)
    profiles["_dose_key"] = profiles["dose"].apply(dose_key)

    dup_mask = profiles.duplicated(["_drug_key", "_dose_key"], keep=False)
    if dup_mask.any():
        examples = profiles.loc[dup_mask, ["pert_iname", "dose"]].drop_duplicates().head(10)
        raise ValueError(
            "The PCA profile table has duplicate drug-dose rows. "
            "Resolve duplicates before pair feature construction.\n"
            f"{examples}"
        )

    lookup = {
        (row["_drug_key"], row["_dose_key"]): row.loc[pc_cols].to_numpy(dtype=float)
        for _, row in profiles.iterrows()
    }

    features: list[np.ndarray] = []
    y_values: list[int] = []
    pair_records: list[dict] = []
    skipped_records: list[dict] = []

    for pair_counter, row in labels.iterrows():
        d1_key = clean_drug_name(row["drug1_matched"])
        d2_key = clean_drug_name(row["drug2_matched"])
        dose1_key = dose_key(row["drug1_conc_num"])
        dose2_key = dose_key(row["drug2_conc_num"])
        key1 = (d1_key, dose1_key)
        key2 = (d2_key, dose2_key)

        missing_reasons = []
        if pd.isna(row["drug1_matched"]):
            missing_reasons.append("drug1_name")
        if pd.isna(row["drug2_matched"]):
            missing_reasons.append("drug2_name")
        if dose1_key is None:
            missing_reasons.append("drug1_conc")
        if dose2_key is None:
            missing_reasons.append("drug2_conc")
        if key1 not in lookup:
            missing_reasons.append("drug1_profile")
        if key2 not in lookup:
            missing_reasons.append("drug2_profile")

        base_record = {
            "pair_id": len(pair_records) + len(skipped_records),
            "label_index": row["label_index"],
            "drug1_original": row["drug1"],
            "drug2_original": row["drug2"],
            "drug1": row["drug1_matched"],
            "drug2": row["drug2_matched"],
            "drug1_conc": row["drug1_conc_num"],
            "drug2_conc": row["drug2_conc_num"],
            "label": row["label_clean"],
            "y": int(row["y"]),
        }

        if missing_reasons:
            skipped = dict(base_record)
            skipped["skip_reason"] = ";".join(missing_reasons)
            skipped_records.append(skipped)
            continue

        features.append(np.concatenate([lookup[key1], lookup[key2]]))
        y_values.append(int(row["y"]))
        pair_records.append(base_record)

    if not features:
        raise ValueError("No drug-pair features could be constructed.")

    pair_meta = pd.DataFrame(pair_records)
    skipped_pairs = pd.DataFrame(skipped_records)
    x_pairs = np.vstack(features)
    y_pairs = np.asarray(y_values, dtype=int)
    return x_pairs, y_pairs, pair_meta, skipped_pairs


def make_model(
    model_name: str,
    random_state: int = 6666,
    rf_n_estimators: int = 200,
    rf_n_jobs: int = -1,
):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=rf_n_estimators,
            random_state=random_state,
            n_jobs=rf_n_jobs,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def run_leave_one_out_predictions(
    x_pairs: np.ndarray,
    y_pairs: np.ndarray,
    pair_meta: pd.DataFrame,
    model_names: Sequence[str] = DEFAULT_MODELS,
    random_state: int = 6666,
    rf_n_estimators: int = 200,
    rf_n_jobs: int = -1,
    threshold: float = 0.5,
) -> pd.DataFrame:
    if x_pairs.shape[0] != y_pairs.shape[0] or x_pairs.shape[0] != pair_meta.shape[0]:
        raise ValueError("X, y, and pair metadata must have the same number of rows.")

    records: list[dict] = []
    loo = LeaveOneOut()
    n_folds = x_pairs.shape[0]
    model_list = list(model_names)
    print(f"Running one fixed leave-one-out pass: {n_folds} folds x {len(model_list)} model(s).")

    for fold_id, (train_idx, test_idx) in enumerate(loo.split(x_pairs)):
        if fold_id == 0 or (fold_id + 1) % 50 == 0 or (fold_id + 1) == n_folds:
            print(f"  LOO fold {fold_id + 1}/{n_folds}")

        x_train, x_test = x_pairs[train_idx], x_pairs[test_idx]
        y_train, y_test = y_pairs[train_idx], y_pairs[test_idx]
        test_row = int(test_idx[0])

        for model_name in model_list:
            model = make_model(
                model_name,
                random_state=random_state,
                rf_n_estimators=rf_n_estimators,
                rf_n_jobs=rf_n_jobs,
            )
            model.fit(x_train, y_train)

            class_to_position = {cls: pos for pos, cls in enumerate(model.classes_)}
            if 1 in class_to_position:
                prob = float(model.predict_proba(x_test)[0, class_to_position[1]])
            else:
                prob = 0.0
            pred = int(prob >= threshold)

            record = pair_meta.iloc[test_row].to_dict()
            record.update(
                {
                    "fold_id": fold_id,
                    "model": model_name,
                    "model_label": MODEL_LABELS.get(model_name, model_name),
                    "y_true": int(y_test[0]),
                    "probability": prob,
                    "prediction": pred,
                    "threshold": threshold,
                }
            )
            records.append(record)

    return pd.DataFrame(records)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc) if not pd.isna(roc_auc) else np.nan,
        "pr_auc": float(pr_auc) if not pd.isna(pr_auc) else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_pairs": int(y_true.shape[0]),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }


def original_metrics_by_model(predictions: pd.DataFrame, source_name: str) -> pd.DataFrame:
    rows: list[dict] = []
    for model_name, model_df in predictions.groupby("model", sort=False):
        metrics = calculate_metrics(
            model_df["y_true"].to_numpy(),
            model_df["prediction"].to_numpy(),
            model_df["probability"].to_numpy(),
        )
        metrics.update(
            {
                "source": source_name,
                "model": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "sample": "original_loo",
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def _bootstrap_indices(
    y_true: np.ndarray,
    rng: np.random.Generator,
    stratified: bool = False,
) -> np.ndarray:
    n = y_true.shape[0]
    if not stratified:
        return rng.choice(n, size=n, replace=True)

    sampled_parts = []
    for label in np.unique(y_true):
        label_idx = np.flatnonzero(y_true == label)
        sampled_parts.append(rng.choice(label_idx, size=label_idx.shape[0], replace=True))
    sampled = np.concatenate(sampled_parts)
    rng.shuffle(sampled)
    return sampled


def _interpolate_roc(y_true: np.ndarray, y_prob: np.ndarray, fpr_grid: np.ndarray) -> np.ndarray:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    tpr_interp[0] = 0.0
    tpr_interp[-1] = 1.0
    return tpr_interp


def _interpolate_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    recall_grid: np.ndarray,
) -> np.ndarray:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    curve = pd.DataFrame({"recall": recall, "precision": precision})
    curve = curve.sort_values("recall").groupby("recall", as_index=False)["precision"].max()
    precision_interp = np.interp(recall_grid, curve["recall"], curve["precision"])
    return np.clip(precision_interp, 0.0, 1.0)


def bootstrap_prediction_metrics(
    predictions: pd.DataFrame,
    source_name: str,
    n_bootstrap: int = 1000,
    random_state: int = 6666,
    curve_grid_size: int = 101,
    stratified: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fpr_grid = np.linspace(0.0, 1.0, curve_grid_size)
    recall_grid = np.linspace(0.0, 1.0, curve_grid_size)
    rng = np.random.default_rng(random_state)

    metric_records: list[dict] = []
    roc_records: list[dict] = []
    pr_records: list[dict] = []
    print(f"Bootstrapping saved out-of-fold prediction rows {n_bootstrap} time(s).")

    for model_name, model_df in predictions.groupby("model", sort=False):
        y_true = model_df["y_true"].to_numpy(dtype=int)
        y_pred = model_df["prediction"].to_numpy(dtype=int)
        y_prob = model_df["probability"].to_numpy(dtype=float)
        model_label = MODEL_LABELS.get(model_name, model_name)

        for bootstrap_id in range(n_bootstrap):
            sample_idx = _bootstrap_indices(y_true, rng, stratified=stratified)
            sample_y = y_true[sample_idx]
            sample_pred = y_pred[sample_idx]
            sample_prob = y_prob[sample_idx]

            metrics = calculate_metrics(sample_y, sample_pred, sample_prob)
            metrics.update(
                {
                    "source": source_name,
                    "model": model_name,
                    "model_label": model_label,
                    "bootstrap_id": bootstrap_id,
                    "stratified": stratified,
                }
            )
            metric_records.append(metrics)

            if len(np.unique(sample_y)) < 2:
                tpr_interp = np.full_like(fpr_grid, np.nan, dtype=float)
                precision_interp = np.full_like(recall_grid, np.nan, dtype=float)
            else:
                tpr_interp = _interpolate_roc(sample_y, sample_prob, fpr_grid)
                precision_interp = _interpolate_pr(sample_y, sample_prob, recall_grid)

            roc_records.extend(
                {
                    "source": source_name,
                    "model": model_name,
                    "model_label": model_label,
                    "bootstrap_id": bootstrap_id,
                    "fpr_grid": float(fpr),
                    "tpr": float(tpr),
                }
                for fpr, tpr in zip(fpr_grid, tpr_interp)
            )
            pr_records.extend(
                {
                    "source": source_name,
                    "model": model_name,
                    "model_label": model_label,
                    "bootstrap_id": bootstrap_id,
                    "recall_grid": float(recall),
                    "precision": float(precision),
                }
                for recall, precision in zip(recall_grid, precision_interp)
            )

    return pd.DataFrame(metric_records), pd.DataFrame(roc_records), pd.DataFrame(pr_records)


def summarize_bootstrap_metrics(
    bootstrap_metrics: pd.DataFrame,
    original_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    original_lookup = {
        row["model"]: row
        for _, row in original_metrics.iterrows()
    }

    for (source, model_name, model_label), group in bootstrap_metrics.groupby(
        ["source", "model", "model_label"], sort=False
    ):
        original = original_lookup.get(model_name)
        for metric in METRIC_COLUMNS:
            values = group[metric].dropna()
            rows.append(
                {
                    "source": source,
                    "model": model_name,
                    "model_label": model_label,
                    "metric": metric,
                    "original_loo": float(original[metric]) if original is not None else np.nan,
                    "bootstrap_mean": float(values.mean()) if not values.empty else np.nan,
                    "bootstrap_sd": float(values.std(ddof=1)) if values.shape[0] > 1 else np.nan,
                    "bootstrap_median": float(values.median()) if not values.empty else np.nan,
                    "bootstrap_q025": float(values.quantile(0.025)) if not values.empty else np.nan,
                    "bootstrap_q975": float(values.quantile(0.975)) if not values.empty else np.nan,
                    "n_bootstrap_valid": int(values.shape[0]),
                }
            )

    return pd.DataFrame(rows)


def run_synergy_analysis(
    source_name: str,
    signature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    gene_cols: Sequence[str],
    output_dir: str | Path,
    n_components: int = 50,
    n_bootstrap: int = 1000,
    random_state: int = 6666,
    model_names: Sequence[str] = DEFAULT_MODELS,
    rf_n_estimators: int = 200,
    rf_n_jobs: int = -1,
    threshold: float = 0.5,
    curve_grid_size: int = 101,
    stratified_bootstrap: bool = False,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)

    profile_features, pca_variance = fit_pca_profile_features(
        signature_df=signature_df,
        gene_cols=gene_cols,
        n_components=n_components,
        random_state=random_state,
    )
    pc_cols = [col for col in profile_features.columns if col.startswith("PC")]

    x_pairs, y_pairs, pair_meta, skipped_pairs = build_pair_feature_table(
        profile_features=profile_features,
        label_df=label_df,
        pc_cols=pc_cols,
    )
    if skipped_pairs.empty:
        skipped_pairs = pd.DataFrame(columns=[*pair_meta.columns, "skip_reason"])

    print(
        f"[{source_name}] Profiles: {signature_df.shape[0]}, "
        f"pairs used: {x_pairs.shape[0]}, skipped pairs: {skipped_pairs.shape[0]}"
    )

    predictions = run_leave_one_out_predictions(
        x_pairs=x_pairs,
        y_pairs=y_pairs,
        pair_meta=pair_meta,
        model_names=model_names,
        random_state=random_state,
        rf_n_estimators=rf_n_estimators,
        rf_n_jobs=rf_n_jobs,
        threshold=threshold,
    )
    original_metrics = original_metrics_by_model(predictions, source_name=source_name)
    bootstrap_metrics, roc_curves, pr_curves = bootstrap_prediction_metrics(
        predictions=predictions,
        source_name=source_name,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        curve_grid_size=curve_grid_size,
        stratified=stratified_bootstrap,
    )
    summary = summarize_bootstrap_metrics(bootstrap_metrics, original_metrics)

    profile_features.to_csv(output_dir / "profile_pca_features.csv", index=True)
    pca_variance.to_csv(output_dir / "pca_explained_variance.csv", index=False)
    pair_meta.to_csv(output_dir / "pair_feature_metadata.csv", index=False)
    skipped_pairs.to_csv(output_dir / "skipped_pairs.csv", index=False)
    predictions.to_csv(output_dir / "loo_predictions.csv", index=False)
    original_metrics.to_csv(output_dir / "original_loo_metrics.csv", index=False)
    bootstrap_metrics.to_csv(output_dir / "bootstrap_metrics.csv", index=False)
    roc_curves.to_csv(output_dir / "roc_curves.csv", index=False)
    pr_curves.to_csv(output_dir / "pr_curves.csv", index=False)
    summary.to_csv(output_dir / "performance_summary.csv", index=False)

    write_json(
        output_dir / "analysis_config.json",
        {
            "source": source_name,
            "n_profiles": int(signature_df.shape[0]),
            "n_gene_columns": int(len(gene_cols)),
            "n_pca_components_requested": int(n_components),
            "n_pca_components_used": int(len(pc_cols)),
            "n_pairs_used": int(x_pairs.shape[0]),
            "n_pairs_skipped": int(skipped_pairs.shape[0]),
            "n_bootstrap": int(n_bootstrap),
            "random_state": int(random_state),
            "models": list(model_names),
            "rf_n_estimators": int(rf_n_estimators),
            "rf_n_jobs": int(rf_n_jobs),
            "threshold": float(threshold),
            "curve_grid_size": int(curve_grid_size),
            "stratified_bootstrap": bool(stratified_bootstrap),
            "bootstrap_unit": "leave-one-out prediction rows",
            "bootstrap_strategy": (
                "Option C: fit PCA once, build pair features once, run LOO once, "
                "then bootstrap out-of-fold prediction rows for metrics and curves."
            ),
            "pca_bootstrapped": False,
            "loo_rerun_within_bootstrap": False,
        },
    )

    print(f"[{source_name}] Wrote outputs to {output_dir}")
    print(summary.to_string(index=False))
    return summary


def compute_pooled_signatures_from_anndata(merged_adata) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_obs = ["pert_iname", "pert_dose", "pert_time", "control", "plate"]
    missing = [col for col in required_obs if col not in merged_adata.obs.columns]
    if missing:
        raise KeyError(f"AnnData obs is missing required columns: {missing}")

    x_matrix = merged_adata.X
    if sparse.issparse(x_matrix):
        x_matrix = x_matrix.toarray()
    expr_df = pd.DataFrame(
        np.asarray(x_matrix),
        index=merged_adata.obs_names,
        columns=merged_adata.var_names,
    )
    obs = merged_adata.obs.copy()

    pert_norm = obs["pert_iname"].astype(str).str.strip().str.upper()
    is_dmso = pert_norm.eq("DMSO").to_numpy()

    ctrl_col = obs["control"]
    if ctrl_col.dtype == bool:
        is_flag_control = ctrl_col.to_numpy()
    else:
        is_flag_control = (
            pd.to_numeric(ctrl_col, errors="coerce").fillna(0).astype(int).eq(1).to_numpy()
        )
    mask_ctrl = np.logical_or(is_flag_control, is_dmso)

    control_means_by_plate = (
        expr_df.loc[mask_ctrl]
        .groupby(obs.loc[mask_ctrl, "plate"], observed=True)
        .mean()
    )
    if control_means_by_plate.empty:
        raise ValueError("No control rows were available to compute plate-centered signatures.")

    plate_series = obs["plate"]
    ctrl_means_aligned = control_means_by_plate.reindex(plate_series)
    if ctrl_means_aligned.isna().any().any():
        missing_plates = plate_series[ctrl_means_aligned.isna().any(axis=1)].dropna().unique()
        raise ValueError(
            "Some plates do not have a matched control mean. "
            f"Missing plates include: {list(missing_plates[:10])}"
        )

    delta_df = pd.DataFrame(
        expr_df.to_numpy(dtype=float) - ctrl_means_aligned.to_numpy(dtype=float),
        index=expr_df.index,
        columns=expr_df.columns,
    )

    mask_treat = ~mask_ctrl
    delta_treat = delta_df.loc[mask_treat]
    grouped = delta_treat.groupby(
        [
            obs.loc[mask_treat, "pert_iname"],
            obs.loc[mask_treat, "pert_dose"],
            obs.loc[mask_treat, "pert_time"],
        ],
        dropna=False,
        observed=True,
    )

    signatures_wide = grouped.mean()
    signatures_wide.index.set_names(["pert_iname", "pert_dose", "pert_time"], inplace=True)
    counts_df = grouped.size().to_frame(name="n_replicates")

    signatures_df = signatures_wide.reset_index()
    meta_cols = ["pert_iname", "pert_dose", "pert_time"]
    gene_cols = [col for col in signatures_df.columns if col not in meta_cols]
    signatures_df = signatures_df.loc[:, [*meta_cols, *gene_cols]]
    return signatures_df, counts_df


def match_reference_to_signatures(
    reference_meta: pd.DataFrame,
    signatures_df: pd.DataFrame,
) -> pd.DataFrame:
    required_ref = ["pert_iname", "dose", "pert_time"]
    required_sig = ["pert_iname", "pert_dose", "pert_time"]
    missing_ref = [col for col in required_ref if col not in reference_meta.columns]
    missing_sig = [col for col in required_sig if col not in signatures_df.columns]
    if missing_ref or missing_sig:
        raise KeyError(
            f"Missing reference columns: {missing_ref}; missing signature columns: {missing_sig}"
        )

    meta_cols_sig = {"pert_iname", "pert_dose", "pert_time", "n_replicates"}
    gene_cols = [col for col in signatures_df.columns if col not in meta_cols_sig]

    ref = reference_meta.copy()
    ref["dose"] = pd.to_numeric(ref["dose"], errors="coerce")
    ref["pert_time"] = pd.to_numeric(ref["pert_time"], errors="coerce")

    sig = signatures_df.copy()
    sig["pert_dose"] = pd.to_numeric(sig["pert_dose"], errors="coerce")
    sig["pert_time"] = pd.to_numeric(sig["pert_time"], errors="coerce")

    sig_by_drug = {
        drug: group.reset_index(drop=True)
        for drug, group in sig.groupby("pert_iname", dropna=False, observed=True)
    }

    out_rows: list[dict] = []
    for _, row in ref.iterrows():
        drug = row["pert_iname"]
        dose_ref = row["dose"]
        time_ref = row["pert_time"]
        candidates = sig_by_drug.get(drug)

        base = {
            "pert_iname_ref": drug,
            "dose_ref": dose_ref,
            "pert_time_ref": time_ref,
            "pert_iname_matched": np.nan,
            "pert_dose_matched": np.nan,
            "pert_time_matched": np.nan,
        }

        if candidates is None or candidates.empty or pd.isna(dose_ref):
            base.update({col: np.nan for col in gene_cols})
            out_rows.append(base)
            continue

        dose_diff = (candidates["pert_dose"] - dose_ref).abs()
        min_diff = dose_diff.min()
        tied = candidates.loc[dose_diff == min_diff]
        max_time = tied["pert_time"].max()
        chosen = tied.loc[tied["pert_time"] == max_time].iloc[0]

        base.update(
            {
                "pert_iname_matched": chosen["pert_iname"],
                "pert_dose_matched": chosen["pert_dose"],
                "pert_time_matched": chosen["pert_time"],
            }
        )
        base.update({col: chosen[col] for col in gene_cols})
        out_rows.append(base)

    matched_df = pd.DataFrame(out_rows)
    front_cols = [
        "pert_iname_ref",
        "dose_ref",
        "pert_time_ref",
        "pert_iname_matched",
        "pert_dose_matched",
        "pert_time_matched",
    ]
    return matched_df.loc[:, [*front_cols, *[col for col in matched_df.columns if col not in front_cols]]]

# %%
"""
Shared utilities for Figure 2 drug-screening revision.

PyCharm-friendly interactive style: run this file first in the Python console,
then run each panel file section-by-section. Figures are saved only; no display.
"""

# %%
from pathlib import Path
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

# %%
BASE_DIR = Path(
    "/Users/meishengxiao/PycharmProjects/PhD_disser/"
    "Experiments/downstreamTaskCode/lung_cancer_A549/DrugRepurposing"
)

TOP_DRUG_DIR = Path(
    "/Users/meishengxiao/PycharmProjects/PhD_disser/"
    "Experiments/downstreamTaskCode/lung_cancer_A549/topDrugNames"
)

DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures" / "figure2"
TABLE_DIR = BASE_DIR / "tables" / "figure2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FINAL_DF_PRED_PATH = TOP_DRUG_DIR / "reverseScore_predicted_All.csv"
FINAL_DF_ORIG_PATH = TOP_DRUG_DIR / "reverseScore_observed_All.csv"
DRUG_META_PATH = Path(
    "/Users/meishengxiao/PycharmProjects/PhD_disser/"
    "data/BroadInst/repurposing_drugs_20200324.txt"
)
GMT_PATH = DATA_DIR / "depict_analysis3_mechanism_panel.v1.Hs.symbols.gmt"

# %%
SOURCE_ORDER = ["Observed LINCS", "DEPICT-predicted LINCS"]
SOURCE_COLORS = {
    "Observed LINCS": "#3B6EA8",
    "DEPICT-predicted LINCS": "#D96C3B",
}

EVIDENCE_ORDER = [
    "Approved / Phase III",
    "NSCLC Phase II",
    "NSCLC Phase I/Ib",
    "General clinical only",
    "NSCLC preclinical",
    "Weak / no evidence",
]
EVIDENCE_COLORS = {
    "Approved / Phase III": "#1B7837",
    "NSCLC Phase II": "#5AAE61",
    "NSCLC Phase I/Ib": "#A6DBA0",
    "General clinical only": "#F4D35E",
    "NSCLC preclinical": "#F4A261",
    "Weak / no evidence": "#BDBDBD",
}

MOA_ORDER = [
    "PI3K–Akt–mTOR",
    "RTK / TKI",
    "p53 / Apoptosis",
    "CDK / Cell cycle",
    "MAPK/MEK/ERK",
    "Non-related / Not found",
]
MOA_COLORS = {
    "PI3K–Akt–mTOR": "#4C78A8",
    "RTK / TKI": "#F58518",
    "p53 / Apoptosis": "#54A24B",
    "CDK / Cell cycle": "#B279A2",
    "MAPK/MEK/ERK": "#E45756",
    "Non-related / Not found": "#9D9D9D",
}

# %%
def set_publication_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

set_publication_style()

# %%
def normalize_drug_name(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    x = x.replace("–", "-").replace("—", "-").replace("−", "-")
    x = re.sub(r"[^a-z0-9]+", "", x)
    aliases = {
        "nvpbez235": "dactolisib",
        "bez235": "dactolisib",
        "dactolisib": "dactolisib",
        "gsk2126458": "omipalisib",
        "omipalisib": "omipalisib",
        "gdc0941": "pictilisib",
        "pictilisib": "pictilisib",
        "mepacrine": "quinacrine",
        "quinacrine": "quinacrine",
        "epoxycholesterol": "epoxycholesterol",
        "brdk26381032": "brdk26381032",
        "brdk18724229": "brdk18724229",
    }
    return aliases.get(x, x)


def load_screening_tables():
    final_df_pred = pd.read_csv(FINAL_DF_PRED_PATH, index_col=0)
    final_df_orig = pd.read_csv(FINAL_DF_ORIG_PATH, index_col=0)
    required = {"pert_iname", "reverse_score"}
    for name, df in {"final_df_pred": final_df_pred, "final_df_orig": final_df_orig}.items():
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
    return final_df_pred, final_df_orig


def load_drug_meta():
    if not DRUG_META_PATH.exists():
        print(f"Drug metadata file not found: {DRUG_META_PATH}")
        return pd.DataFrame()
    return pd.read_csv(DRUG_META_PATH, sep="\t", skiprows=9)


def best_per_drug(df, score_col="reverse_score"):
    tmp = df.dropna(subset=["pert_iname", score_col]).copy()
    idx = tmp.groupby("pert_iname")[score_col].idxmax()
    out = tmp.loc[idx].copy()
    out = out.sort_values(score_col, ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["drug_key"] = out["pert_iname"].map(normalize_drug_name)
    return out


def restrict_predicted_to_observed_drugs(final_df_pred, final_df_orig):
    observed_drugs = final_df_orig["pert_iname"].dropna().unique()
    return final_df_pred[final_df_pred["pert_iname"].isin(observed_drugs)].copy()


def get_top_drug_table(df, source, n=20):
    out = best_per_drug(df).head(n).copy()
    out["source"] = source
    return out

# %%
def manual_annotation_table():
    rows = [
        ("AZD-8330", "AZD-8330", "MEK inhibitor", "MAPK/MEK/ERK", "NSCLC Phase I/Ib"),
        ("MK-2206", "MK-2206", "Akt inhibitor", "PI3K–Akt–mTOR", "NSCLC Phase II"),
        ("PHA-793887", "PHA-793887", "CDK inhibitor", "CDK / Cell cycle", "General clinical only"),
        ("JW-7-24-1", "JW-7-24-1", "Unknown", "Non-related / Not found", "Weak / no evidence"),
        ("ZSTK-474", "ZSTK-474", "Pan PI3K inhibitor", "PI3K–Akt–mTOR", "NSCLC Phase I/Ib"),
        ("PI-828", "PI-828", "Pan PI3K / CK2 inhibitor", "PI3K–Akt–mTOR", "Weak / no evidence"),
        ("foretinib", "Foretinib", "Multikinase TKI", "RTK / TKI", "NSCLC Phase I/Ib"),
        ("PI-103", "PI-103", "PI3K/Akt/mTOR inhibitor", "PI3K–Akt–mTOR", "NSCLC preclinical"),
        ("dasatinib", "Dasatinib", "DDR2/BRAF TKI", "RTK / TKI", "NSCLC Phase II"),
        ("KU-0060648", "KU-0060648", "Dual PI3K/DNA-PK inhibitor", "PI3K–Akt–mTOR", "Weak / no evidence"),
        ("midostaurin", "Midostaurin", "Multikinase inhibitor", "RTK / TKI", "NSCLC Phase I/Ib"),
        ("JWE-035", "JWE-035", "Aurora kinase inhibitor", "Non-related / Not found", "Weak / no evidence"),
        ("NVP-BEZ235", "Dactolisib", "Dual PI3K/mTOR inhibitor", "PI3K–Akt–mTOR", "NSCLC Phase I/Ib"),
        ("dactolisib", "Dactolisib", "Dual PI3K/mTOR inhibitor", "PI3K–Akt–mTOR", "NSCLC Phase I/Ib"),
        ("OSI-027", "OSI-027", "mTOR inhibitor", "PI3K–Akt–mTOR", "General clinical only"),
        ("VU-0418947-2", "VU-0418947-2", "HIF modulator", "Non-related / Not found", "Weak / no evidence"),
        ("CD-437", "CD-437", "Death receptor / apoptosis activator", "p53 / Apoptosis", "NSCLC preclinical"),
        ("torin-2", "Torin-2", "mTOR inhibitor", "PI3K–Akt–mTOR", "NSCLC preclinical"),
        ("MRE-269", "MRE-269", "Prostaglandin I2 receptor agonist", "Non-related / Not found", "Weak / no evidence"),
        ("GSK-2126458", "Omipalisib", "Dual PI3K/mTOR inhibitor", "PI3K–Akt–mTOR", "General clinical only"),
        ("omipalisib", "Omipalisib", "Dual PI3K/mTOR inhibitor", "PI3K–Akt–mTOR", "General clinical only"),
        ("BRD-K26381032", "BRD-K26381032", "Unknown", "Non-related / Not found", "Weak / no evidence"),
        ("BRD-K18724229", "BRD-K18724229", "Unknown", "Non-related / Not found", "Weak / no evidence"),
        ("danusertib", "Danusertib", "Aurora kinase inhibitor", "Non-related / Not found", "General clinical only"),
        ("wortmannin", "Wortmannin", "PI3K inhibitor", "PI3K–Akt–mTOR", "NSCLC preclinical"),
        ("NVP-TAE684", "NVP-TAE684", "ALK inhibitor", "RTK / TKI", "NSCLC preclinical"),
        ("GDC-0941", "Pictilisib", "Pan PI3K inhibitor", "PI3K–Akt–mTOR", "NSCLC Phase I/Ib"),
        ("pictilisib", "Pictilisib", "Pan PI3K inhibitor", "PI3K–Akt–mTOR", "NSCLC Phase I/Ib"),
        ("WYE-125132", "WYE-125132", "mTOR inhibitor", "PI3K–Akt–mTOR", "NSCLC preclinical"),
        ("AZD-7762", "AZD-7762", "Chk1 kinase inhibitor", "CDK / Cell cycle", "NSCLC Phase I/Ib"),
        ("epoxycholesterol", "Epoxy-Cholesterol", "Metabolic regulation", "Non-related / Not found", "Weak / no evidence"),
        ("RITA", "RITA", "p53 activator", "p53 / Apoptosis", "NSCLC preclinical"),
        ("AZD-8055", "AZD-8055", "mTOR inhibitor", "PI3K–Akt–mTOR", "General clinical only"),
        ("dovitinib", "Dovitinib", "FGFR inhibitor", "RTK / TKI", "NSCLC Phase II"),
        ("mepacrine", "Quinacrine", "p53/Wnt/PI3K modulator", "p53 / Apoptosis", "NSCLC Phase I/Ib"),
        ("quinacrine", "Quinacrine", "p53/Wnt/PI3K modulator", "p53 / Apoptosis", "NSCLC Phase I/Ib"),
        ("crizotinib", "Crizotinib", "ALK/ROS1 inhibitor", "RTK / TKI", "Approved / Phase III"),
        ("BMS-754807", "BMS-754807", "IGF-1R inhibitor", "RTK / TKI", "NSCLC preclinical"),
        ("torin-1", "Torin-1", "mTOR inhibitor", "PI3K–Akt–mTOR", "NSCLC preclinical"),
        ("sirolimus", "Sirolimus", "mTOR inhibitor", "PI3K–Akt–mTOR", "NSCLC Phase II"),
        ("TGX-115", "TGX-115", "PI3K inhibitor", "PI3K–Akt–mTOR", "Weak / no evidence"),
    ]
    ann = pd.DataFrame(rows, columns=["pert_iname", "display_name", "moa_label", "moa_group", "evidence_level"])
    ann["drug_key"] = ann["pert_iname"].map(normalize_drug_name)
    return ann.drop_duplicates("drug_key", keep="first")


def annotate_candidates(candidate_df):
    ann = manual_annotation_table()
    out = candidate_df.copy()
    out["drug_key"] = out["pert_iname"].map(normalize_drug_name)
    out = out.merge(
        ann[["drug_key", "display_name", "moa_label", "moa_group", "evidence_level"]],
        on="drug_key",
        how="left",
    )
    out["display_name"] = out["display_name"].fillna(out["pert_iname"])
    out["moa_label"] = out["moa_label"].fillna("Not annotated")
    out["moa_group"] = out["moa_group"].fillna("Non-related / Not found")
    out["evidence_level"] = out["evidence_level"].fillna("Weak / no evidence")
    out["evidence_level"] = pd.Categorical(out["evidence_level"], categories=EVIDENCE_ORDER, ordered=True)
    out["moa_group"] = pd.Categorical(out["moa_group"], categories=MOA_ORDER, ordered=True)
    return out


def build_top20_annotated(restrict_predicted=True):
    final_df_pred, final_df_orig = load_screening_tables()
    if restrict_predicted:
        final_df_pred_for_top = restrict_predicted_to_observed_drugs(final_df_pred, final_df_orig)
    else:
        final_df_pred_for_top = final_df_pred.copy()
    observed_top20 = get_top_drug_table(final_df_orig, "Observed LINCS", n=20)
    predicted_top20 = get_top_drug_table(final_df_pred_for_top, "DEPICT-predicted LINCS", n=20)
    top20 = pd.concat([observed_top20, predicted_top20], ignore_index=True)
    top20 = annotate_candidates(top20)
    suffix = "restricted_predicted" if restrict_predicted else "all_predicted"
    top20.to_csv(TABLE_DIR / f"fig2_top20_annotated_candidates_{suffix}.csv", index=False)
    return top20


def save_figure(fig, stem):
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIG_DIR / (stem + '.pdf')}")

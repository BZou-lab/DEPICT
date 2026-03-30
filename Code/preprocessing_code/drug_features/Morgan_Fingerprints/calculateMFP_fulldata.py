import pandas as pd
import numpy as np
import ast

compounds_info_full = pd.read_csv("./iname_df_update5.csv",index_col=0)

'''
pick out the drugs with only 1 SMILEs and drugs with multiple SMILES
'''
def ensure_list(x):
    """
    Convert a cell to an actual list of SMILES strings.
    Handles:
      • already-a-list            → leave unchanged
      • "['smi1', 'smi2']"       → ast.literal_eval
      • "smi1, smi2"             → split on commas
      • NaN / empty              → []
    """
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        # case 1: looks like a Python list "[ ... ]"
        if x.startswith("[") and x.endswith("]"):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                pass
        # case 2: plain comma-separated string
        return [s.strip().strip("'\"") for s in x.split(",") if s.strip()]
    # fall-back: treat as empty
    return []

# ── 1  standardise the column ───────────────────────────────────────────
compounds_info_full["canonical_smiles_list"] = (
    compounds_info_full["canonical_smiles_list"].apply(ensure_list)
)

# ── 2  count SMILES per drug ────────────────────────────────────────────
compounds_info_full["n_smiles"] = (
    compounds_info_full["canonical_smiles_list"].apply(len)
)

# ── 3  split the table ──────────────────────────────────────────────────
single_smiles_df = compounds_info_full.query("n_smiles == 1").copy()
multi_smiles_df  = compounds_info_full.query("n_smiles  > 1").copy()

print(f"Single-SMILES rows : {len(single_smiles_df):,}")
print(f"Multi-SMILES  rows : {len(multi_smiles_df):,}")

'''
Create the 512-bit Morgan FP for every compound of 17,203 compounds
'''

'''
Majority vote for calculating the Morgan Fingerprints for drugs with multiple SMILES, total 110 drugs

if ceiling(m/2) fingerprints have that bit, then that bit will be set as 1, otherwise set as 0
'''
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfpgen
import numpy as np
from tqdm.auto import tqdm

# ── hyper-parameters ────────────────────────────────────────────────────
RADIUS  = 2
N_BITS  = 512
THRESH  = "majority"            # majority / float p / int k

# 1 build ONE Morgan generator and keep it global
FP_GEN = rfpgen.GetMorganGenerator(
    radius=RADIUS,
    fpSize=N_BITS,          # your 512-bit vector
    includeChirality=False, # leave True if you want stereo
    countSimulation=False   # keep the bits binary
)

# ── helpers ─────────────────────────────────────────────────────────────
def _morgan_bitvect(smiles: str):
    """Return ExplicitBitVect (binary) or None if SMILES fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return FP_GEN.GetFingerprint(mol)          # new API

def _vote_threshold(m: int) -> int:
    if THRESH == "majority":
        return (m + 1) // 2
    if isinstance(THRESH, float):
        return int(np.ceil(m * THRESH))
    return int(THRESH)

def majority_vote_fp(smiles_list):
    fps = []
    for smi in smiles_list:
        fp = _morgan_bitvect(smi)
        if fp is not None:
            arr = np.zeros((N_BITS,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)

    m = len(fps)
    if m == 0:
        return None

    votes = np.sum(np.vstack(fps), axis=0)
    k     = _vote_threshold(m)
    return (votes >= k).astype(np.uint8)

# ── progress-bar-enhanced apply ─────────────────────────────────────────
tqdm.pandas(desc="Majority-vote FP")

multi_smiles_df["fp"] = (
    multi_smiles_df["canonical_smiles_list"]
    .progress_apply(majority_vote_fp)
)

multi_smiles_df.dropna(subset=["fp"], inplace=True)
print(f"Finished: {len(multi_smiles_df):,} rows now have a majority-vote fingerprint.")

# ── 1. make sure we’re working with rows that *have* a fingerprint ─────
multi_ready = multi_smiles_df.dropna(subset=["fp"]).copy()

# ── 2. convert the Series of uint8 arrays → 2-D NumPy matrix ───────────
fp_mat = np.vstack(multi_ready["fp"].values)          # shape = (n_drugs, 512)

# ── 3. build the bit-level DataFrame ───────────────────────────────────
bit_cols = list(range(fp_mat.shape[1]))               # [0, 1, …, 511]
multi_bits_df = pd.DataFrame(
    fp_mat,
    index=multi_ready["pert_iname"],                  # row index
    columns=bit_cols
).astype("uint8")                                     # keeps it tiny

# optional: check uniqueness of the index
if not multi_bits_df.index.is_unique:
    print("Warning: duplicate pert_iname values detected!")

'''
Calculate the MFP for the rest 17,091 drugs with single SMILES
'''
# ── 0.  progress-bar wrapper for apply ──────────────────────────────────
tqdm.pandas(desc="Single-SMILES FP")

# ── 1.  build a binary Morgan FP for each row (k = 1) ───────────────────
def single_fp(smiles_list):
    """Generate a binary uint8 vector for exactly one SMILES."""
    if not smiles_list:                       # empty list or NaN → skip
        return None
    fp = _morgan_bitvect(smiles_list[0])      # first/only SMILES
    if fp is None:
        return None
    arr = np.zeros((N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

single_smiles_df["fp"] = (
    single_smiles_df["canonical_smiles_list"]
    .progress_apply(single_fp)
)

single_smiles_df.dropna(subset=["fp"], inplace=True)

# ── 2.  convert the fp column → bit-matrix DataFrame ────────────────────
fp_mat_single = np.vstack(single_smiles_df["fp"].values)   # (n_single, 512)
bit_cols      = list(range(N_BITS))                        # 0 … 511

single_bits_df = pd.DataFrame(
    fp_mat_single,
    index=single_smiles_df["pert_iname"],
    columns=bit_cols
).astype("uint8")

print(f"Single-SMILES bit-matrix shape: {single_bits_df.shape}")
print(single_bits_df.head())

all_bits_df = pd.concat([single_bits_df, multi_bits_df], axis=0)

all_bits_df.to_csv('./compounds_512MFP_wholeDat.csv')
# remember to use index_col = 0, when read in data.
all_bits_df_read = pd.read_csv('./compounds_512MFP_wholeDat.csv', index_col=0)

# there are two drugs missing after calculating the MFP
# isosorbide-mononitrate
# rigosertib

isosorbide_mo = compounds_info_full[compounds_info_full['pert_iname']=='isosorbide-mononitrate']
rigosertib = compounds_info_full[compounds_info_full['pert_iname']=='rigosertib']

# these two drugs are missing their CID and SMILES

# now got the fixed data frame, and calculate the two drugs MFP
compounds_info_fixed = pd.read_csv("./compounds_df_wTarMoA_full_fixed.csv",index_col=0) # this dataset was manually fixed from the 'compounds_512MFP_wholeDat.csv'.

mask = compounds_info_fixed["pert_iname"].isin(
    ["isosorbide-mononitrate", "rigosertib"]
)

subset = compounds_info_fixed[mask].copy()

# ── 1  standardise the column ───────────────────────────────────────────
subset["canonical_smiles_list"] = (
    subset["canonical_smiles_list"].apply(ensure_list)
)

# ── 2  count SMILES per drug ────────────────────────────────────────────
subset["n_smiles"] = (
    subset["canonical_smiles_list"].apply(len)
)


# ── 0.  progress-bar wrapper for apply ──────────────────────────────────
tqdm.pandas(desc="Single-SMILES FP")

# ── 1.  build a binary Morgan FP for each row (k = 1) ───────────────────
def single_fp(smiles_list):
    """Generate a binary uint8 vector for exactly one SMILES."""
    if not smiles_list:                       # empty list or NaN → skip
        return None
    fp = _morgan_bitvect(smiles_list[0])      # first/only SMILES
    if fp is None:
        return None
    arr = np.zeros((N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

subset["fp"] = (
    subset["canonical_smiles_list"]
    .progress_apply(single_fp)
)

subset.dropna(subset=["fp"], inplace=True)

# ── 2.  convert the fp column → bit-matrix DataFrame ────────────────────
fp_mat_single = np.vstack(subset["fp"].values)   # (n_single, 512)
bit_cols      = list(range(N_BITS))                        # 0 … 511

miss2drugs_bits_df = pd.DataFrame(
    fp_mat_single,
    index=subset["pert_iname"],
    columns=bit_cols
).astype("uint8")

print(f"Single-SMILES bit-matrix shape: {miss2drugs_bits_df.shape}")
print(miss2drugs_bits_df.head())


all_bits_df_read.columns  = all_bits_df_read.columns.astype(int)
miss2drugs_bits_df.columns = miss2drugs_bits_df.columns.astype(int)

all_bits_df_fixed = pd.concat([all_bits_df_read, miss2drugs_bits_df], axis=0)

all_bits_df_fixed.to_csv('./Data/FinalData/compounds_512MFP_wholeDat_fixed.csv')
# remember to use index_col = 0, when read in data.
# all_bits_df_fixed_read = pd.read_csv('/Users/meishengxiao/PycharmProjects/PhD_disser/data/LINCS_PRNet/final_data/compounds_512MFP_wholeDat_fixed.csv', index_col=0)

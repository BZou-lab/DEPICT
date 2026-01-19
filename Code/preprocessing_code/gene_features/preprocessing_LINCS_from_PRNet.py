'''
Created by Meisheng Xiao, Jul 2025
Last checked and modified at Jan 2026, for creating a GitHub Version
'''
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split

'''
Please download the above raw data before running the code

LINCS L1000 from PRNet: https://zenodo.org/records/14230870
'''

# this file is not in the folder, please download it before running the preprocessing code
adata = sc.read('.../DEPICT/Data/RawData/Lincs_L1000.h5ad') # For READERS: please assign your own path.

'''
CONCENTRATION AND DURATION

Jul3: I think we dont need to preprocess this, it is clean for the PRNet data,
      only do sanity check on the quantity and units.

For readers: You can skip this part, just a quick sanity check for the raw dataset
'''
adata.obs['dose'].nunique() # 2,889 dosage conditions
perturbagen_dose_tab = adata.obs['dose'].value_counts()

adata.obs['pert_dose_unit'].nunique() # 3 dose unit
perturbagen_dose_unit_tab = adata.obs['pert_dose_unit'].value_counts() # either uM or -666/%, there are 46,396 not in uM, guess it should be DMSO.

adata.obs['pert_iname'].nunique() # 17,203 perturbagens
perturbagen_tab = adata.obs['pert_iname'].value_counts() # 46,428 DMSO, why there are 46428 - 46396 = 32 DMSO with other dose unit?
# now check why some DMSO have other dose unit.
DMSO_sub = adata[adata.obs['pert_iname'] == 'DMSO'].copy()
DMSO_dose_unit_tab = DMSO_sub.obs['pert_dose_unit'].value_counts() # 32 are in um dose unite
DMSO_dose_tab = DMSO_sub.obs['dose'].value_counts() # all DMSO has dose 0.
# but there are still other perturbagens have 0 dose, check which
Odose_sub = adata[adata.obs['dose'] == 0].copy()
Odose_perturbagen_tab = Odose_sub.obs['pert_iname'].value_counts() # yes, there are some perturbagens than DMSO have 0 dose, IDK why.

adata.obs['pert_time'].nunique() # 4 time points
perturbagen_time_tab = adata.obs['pert_time'].value_counts() # 543,999 24h; 325,164 6h; 13,676 3h; 430 48h.
# 98.4% ((543999 + 325164) / 883269) experiments are in 24-h or 6-h duration;
# so when modeling the whole dataset, we can have a module control the duration
# or we can train on the 24-h and 6-h separately, without the data on 3-h and 48-h

adata.obs['pert_time_unit'].nunique() # 1 time unit
perturbagen_time_unit_tab = adata.obs['pert_time_unit'].value_counts() # all in hour as time unit

'''
ASSIGN CONTROL PAIR TO EACH EXPERIMENT

randomly assign a DMSO sample to each of samples in the same plate.

Note that there are 'det_plate' and 'rna_plate' in the data and all indicates the plate info. Be care.
'''
adata.obs['det_plate'].nunique() # 916 det plates, total 204,676 experiments, 678,593 nan
perturbagen_det_plate_tab = adata.obs['det_plate'].value_counts()

adata.obs['rna_plate'].nunique() # 1985 rna plates, total 678,593 experiments, 204,676 nan
perturbagen_rna_plate_tab = adata.obs['rna_plate'].value_counts()

## first check for every plate, it has DMSO or not.
# Create a new column 'plate' that merges 'det_plate' and 'rna_plate'
# Convert string 'nan' to actual NaN
# Convert 'det_plate' and 'rna_plate' to string, ensuring 'nan' is replaced with actual NaN
adata1 = adata.copy()
adata1.obs['det_plate'] = adata1.obs['det_plate'].astype(str).replace('nan', np.nan)
adata1.obs['rna_plate'] = adata1.obs['rna_plate'].astype(str).replace('nan', np.nan)
# check
adata1_rna_plate_tab = adata1.obs['rna_plate'].value_counts()

# Now merge into a new 'plate' column
adata1.obs['plate'] = adata1.obs['det_plate'].fillna(adata1.obs['rna_plate'])

# check
check_plate_info = adata1.obs[['det_plate', 'rna_plate', 'plate']]
plate_tab = adata1.obs['plate'].value_counts() # 1984 + 915 = 2899, total number of plates is right.
plate_tab.sum() # 883,269, total number of experiments is right

# Step 1: Get all unique plates
all_plates = adata1.obs['plate'].unique()

# Step 2: Find plates where 'DMSO' exists
plates_with_dmso = adata1.obs[adata1.obs['pert_iname'] == 'DMSO']['plate'].unique()

# Step 3: Find missing plates
missing_plates = set(all_plates) - set(plates_with_dmso) # 24 out of 2899 plates do not have DMSO on them.

# Step 4: Print results
if not missing_plates:
    print("✅ DMSO is present in every plate!")
else:
    print("❌ DMSO is missing in the following plates:", missing_plates)

mask = adata1.obs["plate"].isin(missing_plates)   # boolean mask for rows whose plate is in missing_plates
adata_missing = adata1[mask].copy() # just only 192 experiments in these plates

drug_tab_adataMissingDMSO = adata_missing.obs['pert_iname'].value_counts()
adata_missing.obs['pert_type'].unique()

adata_missing.obs['paired_control_index'].nunique()
control_tab_adataMissingDMSO = adata_missing.obs['paired_control_index'].value_counts()

duration_tab_adataMissingDMSO = adata_missing.obs['pert_time'].value_counts()
# strange thing is that these are 48h experiments but assigned with 6 h and 24 h baselines
cell_tab_adataMissingDMSO = adata_missing.obs['cell_id'].value_counts() # cell is HEK293T
'''
for those plates with no DMSOs, if there are baselines with the same cell and same duration, we can assign them.
If there are not, then delete these plates
'''
mask = (adata1.obs["cell_id"] == "HEK293T") & (adata1.obs["pert_time"] == 48)
adata_hek48 = adata1[mask].copy()
adata_hek48.X == adata_missing.X # exactly the same
# the HEK293T cell and 48 hours does not have any DMSO control.
# delete these plates without any DMSO controls on it.
mask = ~adata1.obs["plate"].isin(missing_plates)
adata2 = adata1[mask].copy()      # keep everything else



'''
randomly assign a DMSO sample to each of samples in the same plate
'''
adata2.obs.columns
adata2_DMSOsub = adata2[adata2.obs['pert_iname']=='DMSO']
DMSO_control_count = adata2_DMSOsub.obs['control'].value_counts()
adataSubset_control_count = adata2.obs['control'].value_counts()

# already every DMSO as control 1 and others as control 0
# Step 1: Assign control == 1 for DMSO and control == 0 for others
# adata_subset.obs['control'] = np.where(adata_subset.obs['pert_iname'] == 'DMSO', 1, 0)

# Step 2: Initialize 'paired_control_index' with NaN
adata2.obs['paired_control_index'] = np.nan

# random_row_adata.obs.index # what is this for?

# Step 2: Initialize 'paired_control_index' as an empty string column (avoiding float dtype)
adata2.obs['paired_control_index'] = ""

# Step 3: Group samples by plate and assign a DMSO index to each non-DMSO sample
for plate in adata2.obs['plate'].unique():
    # Get indices of DMSO samples in this plate
    dmso_indices = adata2.obs[
        (adata2.obs['plate'] == plate) & (adata2.obs['control'] == 1)].index.astype(str)

    # Get indices of non-DMSO samples in this plate
    non_dmso_indices = adata2.obs[(adata2.obs['plate'] == plate) & (adata2.obs['control'] == 0)].index

    # Step 4: Randomly assign a DMSO index to each non-DMSO sample
    if len(dmso_indices) > 0 and len(non_dmso_indices) > 0:
        paired_indices = np.random.choice(dmso_indices, size=len(non_dmso_indices), replace=True).astype(str)

        # Assign as strings to avoid dtype mismatch
        adata2.obs.loc[non_dmso_indices, 'paired_control_index'] = paired_indices

# Step 5: Verify dtype and values
print(adata2.obs[['plate', 'pert_iname', 'control', 'paired_control_index']].head(20))
check_control = adata2.obs[['plate', 'pert_iname', 'control', 'paired_control_index']].head(20)
print("Final dtype of paired_control_index:", adata2.obs['paired_control_index'].dtype)

'''
check the first row correct pair or not
correct pair
'''
adata_subset_DMSOsub = adata2[adata2.obs['pert_iname']=='DMSO']
check_control_DMSO = adata_subset_DMSOsub.obs[['plate', 'pert_iname', 'control', 'paired_control_index']].head(20)
adata2.obs[['paired_control_index']].head(20)
# Extract the first paired_control_index value
first_paired_index = adata2.obs['paired_control_index'].iloc[15]
paired_row = adata2[first_paired_index]
paired_row.obs['plate']
adata2.obs['plate'].iloc[15] # same plate
paired_row.obs['pert_iname'] # DMSO
adata2.obs['pert_iname'].iloc[15] # a drug's name

#### check if every experiment rather than controls have a paired index or not
adata2_nonDMSO = adata2[adata2.obs['pert_iname']!='DMSO'].copy()

# ── 1. identify rows where `paired_control_index` is *empty* or *NaN* ─────────
is_empty = adata2_nonDMSO.obs["paired_control_index"].eq("")          # ""
is_na    = adata2_nonDMSO.obs["paired_control_index"].isna()          # NaN/None
missing_mask = is_empty | is_na

n_missing = missing_mask.sum()
print(f"{n_missing} of {adata2_nonDMSO.n_obs} rows lack a paired control index")

# ── 2. quick boolean answer: “Does every row have a non-empty value?” ─────────
all_filled = n_missing == 0
print("Every row has a non-empty paired_control_index:", all_filled)
#### all experiments have a paired index.
'''
Because the data is large, if your machine cannot ram cannot handle. save it and clear space and load it again.
'''
# adata2.write(".../DEPICT/Data/RawData/halfway_data/adata2.h5ad")
#
# # check difference
# adata2_read = sc.read(".../DEPICT/Data/RawData/halfway_data/adata2.h5ad")
# # all good.
#
# adata2 = sc.read(".../DEPICT/Data/RawData/halfway_data/adata2.h5ad")

'''
CREATE SPLIT STRATEGY FOR TRAINING, VALIDATION AND TEST

have a split for random, cell ,drug, cell&drug split
first have a subset without DMSO, and split the subset
after that concatenate the subset with the DMSO subset
'''
adata2.obs.columns
columns_to_remove = [
    'cell_type_split_0', 'cell_type_split_1', 'cell_type_split_2',
    'cell_type_split_3', 'cell_type_split_4', 'random_split_0',
    'random_split_1', 'random_split_2', 'random_split_3', 'random_split_4',
    'drug_split_0', 'drug_split_1', 'drug_split_2', 'drug_split_3',
    'drug_split_4', 'cov_drug_dose_name_split_0',
    'cov_drug_dose_name_split_1', 'cov_drug_dose_name_split_2',
    'cov_drug_dose_name_split_3', 'cov_drug_dose_name_split_4'
]
adata2.obs = adata2.obs.drop(columns=columns_to_remove)

adata_dmso = adata2[adata2.obs['pert_iname'] == 'DMSO'].copy()

# Subset where 'pert_iname' is not 'DMSO'
adata_non_dmso = adata2[adata2.obs['pert_iname'] != 'DMSO'].copy()


# Function to split a given array into train (80%), valid (10%), and test (10%)
def split_data(unique_values):
    train, temp = train_test_split(unique_values, test_size=0.2, random_state=666)
    valid, test = train_test_split(temp, test_size=0.5, random_state=666)
    return train, valid, test

# Initialize new obs columns
adata_dmso.obs['random_split'] = ''
adata_dmso.obs['cell_split'] = ''
adata_dmso.obs['drug_split'] = ''
# Initialize new obs columns
adata_non_dmso.obs['random_split'] = ''
adata_non_dmso.obs['cell_split'] = ''
adata_non_dmso.obs['drug_split'] = ''

# 1. Random Split
train_idx, valid_idx, test_idx = split_data(adata_non_dmso.obs.index)
adata_non_dmso.obs.loc[train_idx, 'random_split'] = 'train'
adata_non_dmso.obs.loc[valid_idx, 'random_split'] = 'valid'
adata_non_dmso.obs.loc[test_idx, 'random_split'] = 'test'

# 2. Cell Split
unique_cells = adata_non_dmso.obs['cell_id'].unique()
train_cells, valid_cells, test_cells = split_data(unique_cells)

adata_non_dmso.obs.loc[adata_non_dmso.obs['cell_id'].isin(train_cells), 'cell_split'] = 'train'
adata_non_dmso.obs.loc[adata_non_dmso.obs['cell_id'].isin(valid_cells), 'cell_split'] = 'valid'
adata_non_dmso.obs.loc[adata_non_dmso.obs['cell_id'].isin(test_cells), 'cell_split'] = 'test'

# 3. Drug Split
unique_drugs = adata_non_dmso.obs['pert_iname'].unique()
train_drugs, valid_drugs, test_drugs = split_data(unique_drugs)

adata_non_dmso.obs.loc[adata_non_dmso.obs['pert_iname'].isin(train_drugs), 'drug_split'] = 'train'
adata_non_dmso.obs.loc[adata_non_dmso.obs['pert_iname'].isin(valid_drugs), 'drug_split'] = 'valid'
adata_non_dmso.obs.loc[adata_non_dmso.obs['pert_iname'].isin(test_drugs), 'drug_split'] = 'test'

# ────────────────────────────────────────────────────────────────────────────────
def add_triple_splits(
    adata,
    seeds=(6, 66, 666, 6666, 66666),
    train_frac=0.80, valid_frac=0.10, test_frac=0.10,
):
    """
    For every seed in `seeds`, append three new obs-columns:
      * random_split{n}
      * cell_split{n}
      * drug_split{n}

    Parameters
    ----------
    adata : AnnData
        The (non-DMSO) AnnData object whose .obs table you want to augment.
    seeds : iterable[int], default (666, …, 670)
        Five random seeds → five parallel splits per strategy.
    train_frac, valid_frac, test_frac : float
        Must sum to 1.0.  Defaults: 0.80 / 0.10 / 0.10.
    """
    assert abs(train_frac + valid_frac + test_frac - 1.0) < 1e-9, "fractions must sum to 1"

    def _split_once(values, seed):
        """Return three mutually exclusive arrays according to the given fractions."""
        train, temp = train_test_split(values, test_size=valid_frac + test_frac, random_state=seed)
        # scale the remaining (validation+test) fraction so that test_frac is correct
        rel_test = test_frac / (valid_frac + test_frac)
        valid, test = train_test_split(temp, test_size=rel_test, random_state=seed)
        return train, valid, test

    # main loop over seeds
    for idx, seed in enumerate(seeds, start=1):
        # ---- strategy 1: purely random (cell-level) --------------------------------
        train_idx, valid_idx, test_idx = _split_once(adata.obs.index.values, seed)
        col = f"random_split{idx}"
        adata.obs[col] = ""              # initialise column
        adata.obs.loc[train_idx, col] = "train"
        adata.obs.loc[valid_idx, col] = "valid"
        adata.obs.loc[test_idx,  col] = "test"

        # ---- strategy 2: by cell line ---------------------------------------------
        unique_cells = adata.obs["cell_id"].unique()
        train_cells, valid_cells, test_cells = _split_once(unique_cells, seed)
        col = f"cell_split{idx}"
        adata.obs[col] = ""
        adata.obs.loc[adata.obs["cell_id"].isin(train_cells), col] = "train"
        adata.obs.loc[adata.obs["cell_id"].isin(valid_cells), col] = "valid"
        adata.obs.loc[adata.obs["cell_id"].isin(test_cells),  col] = "test"

        # ---- strategy 3: by drug ---------------------------------------------------
        unique_drugs = adata.obs["pert_iname"].unique()
        train_drugs, valid_drugs, test_drugs = _split_once(unique_drugs, seed)
        col = f"drug_split{idx}"
        adata.obs[col] = ""
        adata.obs.loc[adata.obs["pert_iname"].isin(train_drugs), col] = "train"
        adata.obs.loc[adata.obs["pert_iname"].isin(valid_drugs), col] = "valid"
        adata.obs.loc[adata.obs["pert_iname"].isin(test_drugs),  col] = "test"

    return adata  # (returned for convenience, but function mutates in place)

# ───── usage ────────────────────────────────────────────────────────────────────
adata_non_dmso = add_triple_splits(adata_non_dmso)  # uses default five seeds

# Initialize new obs columns
adata_dmso.obs['random_split1'] = ''
adata_dmso.obs['cell_split1'] = ''
adata_dmso.obs['drug_split1'] = ''
adata_dmso.obs['random_split2'] = ''
adata_dmso.obs['cell_split2'] = ''
adata_dmso.obs['drug_split2'] = ''
adata_dmso.obs['random_split3'] = ''
adata_dmso.obs['cell_split3'] = ''
adata_dmso.obs['drug_split3'] = ''
adata_dmso.obs['random_split4'] = ''
adata_dmso.obs['cell_split4'] = ''
adata_dmso.obs['drug_split4'] = ''
adata_dmso.obs['random_split5'] = ''
adata_dmso.obs['cell_split5'] = ''
adata_dmso.obs['drug_split5'] = ''


# Verify the splits
adata_non_dmso.obs.columns
adata_dmso.obs.columns
# split 1
check_random_split1 = adata_non_dmso.obs[['random_split1']].value_counts() # train:test:valid = 669319 : 83665 : 83665
check_cell_split1 = adata_non_dmso.obs[['cell_split1']].value_counts() # train:test:valid = 633012 : 12510 : 191127
check_drug_split1 = adata_non_dmso.obs[['drug_split1']].value_counts() # train:test:valid = 660833 : 85539 : 90277
# split 2
check_random_split2 = adata_non_dmso.obs[['random_split2']].value_counts() # train:test:valid = 669319 : 83665 : 83665
check_cell_split2 = adata_non_dmso.obs[['cell_split2']].value_counts() # train:test:valid = 730869 : 25742 : 80038
check_drug_split2 = adata_non_dmso.obs[['drug_split2']].value_counts() # train:test:valid = 667131 : 86935 : 82583
# split 3
check_random_split3 = adata_non_dmso.obs[['random_split3']].value_counts() # train:test:valid = 669319 : 83665 : 83665
check_cell_split3 = adata_non_dmso.obs[['cell_split3']].value_counts() # train:test:valid = 619257 : 156258 : 61134
check_drug_split3 = adata_non_dmso.obs[['drug_split3']].value_counts() # train:test:valid = 670367 : 78097 : 88185
# split 4
check_random_split4 = adata_non_dmso.obs[['random_split4']].value_counts() # train:test:valid = 669319 : 83665 : 83665
check_cell_split4 = adata_non_dmso.obs[['cell_split4']].value_counts() # train:test:valid = 671357 : 137725 : 27567
check_drug_split4 = adata_non_dmso.obs[['drug_split4']].value_counts() # train:test:valid = 675250 : 77889 : 83510
# split 5
check_random_split5 = adata_non_dmso.obs[['random_split5']].value_counts() # train:test:valid = 669319 : 83665 : 83665
check_cell_split5 = adata_non_dmso.obs[['cell_split5']].value_counts() # train:test:valid = 590631 : 86370 : 159648
check_drug_split5 = adata_non_dmso.obs[['drug_split5']].value_counts() # train:test:valid = 661871 : 80260 : 94518

adata_combined = ad.concat([adata_non_dmso, adata_dmso], join="outer")

check_random_split_combined = adata_combined.obs[['random_split1']].value_counts()
check_cell_split_combined = adata_combined.obs[['cell_split1']].value_counts()
check_drug_split_combined = adata_combined.obs[['drug_split1']].value_counts()


adata_combined.write(".../DEPICT/Data/FinalData/halfway_data/adataAfterClean.h5ad") # this is the data we will be using at training, validation and inference.
# demographics for the used data
adata_combined.obs['cell_id'].nunique()
adata_combined.obs['pert_iname'].nunique()
adata_combined.obs['plate'].nunique()
adata_combined.obs['control'].value_counts()

'''
Loewe score calculation, to determine Synergistic or Antagonistic
'''
import pandas as pd
import scanpy as sc
import numpy as np
from synergy.combination.loewe import Loewe
from synergy.single.hill import Hill

HT29_doublet = pd.read_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_pairData.csv", index_col=0)
HT29_single = pd.read_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_singleData.csv", index_col=0)


# ------------------------
# Step 2. Convert to inhibition
# ------------------------
def to_inhibition(x):
    """Convert X/X0 viability to inhibition and clip to [0,1]."""
    return np.clip(1 - x, 0, 1)

HT29_single["inhibition"] = to_inhibition(HT29_single["X/X0"])
HT29_doublet["inhibition"] = to_inhibition(HT29_doublet["X/X0"])

# ------------------------
# Step 3. Function for one drug pair
# ------------------------
def compute_loewe_for_pair(drugA, drugB, single_df, combo_df):
    """
    Compute Loewe CI for one drug pair (across all doses).
    Returns a DataFrame with per-dose CI and categorical label.
    """

    # --- Extract single-agent data
    singles_A = single_df[single_df["drug_name"] == drugA][["Drug_concentration (µM)", "inhibition"]].copy()
    singles_A = singles_A.rename(columns={"Drug_concentration (µM)": "drug1_conc"})
    singles_A["drug2_conc"] = 0.0

    singles_B = single_df[single_df["drug_name"] == drugB][["Drug_concentration (µM)", "inhibition"]].copy()
    singles_B = singles_B.rename(columns={"Drug_concentration (µM)": "drug2_conc"})
    singles_B["drug1_conc"] = 0.0

    # --- Extract combination data
    combos = combo_df[
        (combo_df["drugA_name"] == drugA) & (combo_df["drugB_name"] == drugB)
    ][["drugA Conc (µM)", "drugB Conc (µM)", "inhibition"]].copy()
    combos = combos.rename(columns={"drugA Conc (µM)": "drug1_conc",
                                    "drugB Conc (µM)": "drug2_conc"})

    # --- Merge into tidy dataset
    df = pd.concat([
        singles_A[["drug1_conc", "drug2_conc", "inhibition"]],
        singles_B[["drug1_conc", "drug2_conc", "inhibition"]],
        combos[["drug1_conc", "drug2_conc", "inhibition"]]
    ], ignore_index=True)

    # --- Fit Loewe model
    model = Loewe(mode="ci", drug1_model=Hill, drug2_model=Hill)
    model.fit(df["drug1_conc"].values,
              df["drug2_conc"].values,
              df["inhibition"].values)

    CI = model.synergy  # Combination Index values

    # --- Attach results back to df
    df["CI"] = CI
    df["label"] = df["CI"].apply(lambda x: "synergy" if x < 1 else ("additive" if x == 1 else "antagonism"))

    # --- Add drug names
    df["drug1"] = drugA
    df["drug2"] = drugB

    return df

# ------------------------
# Step 4. Loop over all pairs
# ------------------------
results = []

unique_pairs = HT29_doublet[["drugA_name", "drugB_name"]].drop_duplicates()

for _, row in unique_pairs.iterrows():
    drugA, drugB = row["drugA_name"], row["drugB_name"]

    try:
        df_pair = compute_loewe_for_pair(drugA, drugB, HT29_single, HT29_doublet)
        results.append(df_pair)
    except Exception as e:
        print(f"Skipping {drugA}+{drugB} due to error: {e}")

# Combine all into one DataFrame
results_df = pd.concat(results, ignore_index=True)
'''
CI == nan, meaning this is from extrapolation so prohibited, as no effect defined in this effect area in single-agent data.
pick out those doublet observations.
'''
# Drop rows where CI is NaN or exactly 1
filtered_df = results_df.dropna(subset=["CI"])
combo_only_df = filtered_df[(filtered_df["drug1_conc"] != 0) & (filtered_df["drug2_conc"] != 0)]

combo_only_df['label'].value_counts() # 357 antagonism and 199 synergetic.

# ------------------------
# Step 5. Save output
# ------------------------
combo_only_df.to_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_allpairs_LoeweCI_labels.csv", index=False)
print("Done! Results saved to HT29_allpairs_LoeweCI_labels.csv")

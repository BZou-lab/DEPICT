import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

compounds_info = pd.read_csv("./compounds_df_wTarMoA_full_fixed.csv",index_col=0)


# assume compounds_info already exists
# ------------------------------------------------------------------
# 1. Split the pipe-separated targets into lists
compounds_info["target_list"] = (
    compounds_info["target"]
    .fillna("")                # handle NaNs
    .str.split("|")            # → list[str]
    .apply(lambda lst: [t for t in lst if t])   # drop empty tokens
)

# 2. Fit a MultiLabelBinarizer to all target labels
mlb = MultiLabelBinarizer()
target_matrix = mlb.fit_transform(compounds_info["target_list"])

# 3. Turn the matrix into a DataFrame with one column per target
target_df = pd.DataFrame(
    target_matrix,
    index=compounds_info["pert_iname"],   # ← use pert_iname as index
    columns=mlb.classes_
)


target_df.to_csv("./compounds_target_multihot_full.csv")

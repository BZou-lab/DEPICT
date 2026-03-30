import pandas as pd
import numpy as np

df_spearman_orig = pd.read_csv("./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/df_spearman_meanPlate.csv")
df_spearman_pred = pd.read_csv("./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/df_spearman_pred_meanPlate.csv")


df_connectivity_orig = pd.read_csv("./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/df_connectivity_meanPlate.csv")
df_connectivity_pred = pd.read_csv("./Code/downstream_analysis_code/DrugPrioritization/NSCLC_data/df_connectivity_pred_meanPlate.csv")


final_df_orig = (
    df_connectivity_orig[['obs_name','CS_raw','pert_iname','pert_dose','pert_time']]
    .merge(df_spearman_orig[['obs_name','spearman_r','pert_iname','pert_dose','pert_time']],
           on='obs_name', suffixes=('_c','_s'))
    .assign(reverse_score=lambda d: pd.to_numeric(d['CS_raw'], errors='coerce') +
                                    pd.to_numeric(d['spearman_r'], errors='coerce'))
    .rename(columns={'pert_iname_c':'pert_iname','pert_dose_c':'pert_dose','pert_time_c':'pert_time'})
    [['obs_name','reverse_score','pert_iname','pert_dose','pert_time']]
    .sort_values('reverse_score', ascending=False)
    .reset_index(drop=True)
)


final_df_pred = (
    df_connectivity_pred[['obs_name','CS_raw','pert_iname','pert_dose','pert_time']]
    .merge(df_spearman_pred[['obs_name','spearman_r','pert_iname','dose','pert_time']],
           on='obs_name', suffixes=('_c','_s'))
    .assign(reverse_score=lambda d: pd.to_numeric(d['CS_raw'], errors='coerce') +
                                    pd.to_numeric(d['spearman_r'], errors='coerce'))
    .rename(columns={'pert_iname_c':'pert_iname','pert_dose_c':'pert_dose','pert_time_c':'pert_time'})
    [['obs_name','reverse_score','pert_iname','pert_dose','pert_time']]
    .sort_values('reverse_score', ascending=False)
    .reset_index(drop=True)
)


# ensure numeric
final_df_orig['reverse_score'] = pd.to_numeric(final_df_orig['reverse_score'], errors='coerce')

# 1) pick the row with the max reverse_score within each pert_iname
best_idx = final_df_orig.groupby('pert_iname')['reverse_score'].idxmax()
top_per_drug = final_df_orig.loc[best_idx]

# 2) sort those by reverse_score and take the top 20
top20 = (top_per_drug
         .sort_values('reverse_score', ascending=False)
         .head(20)
         .reset_index(drop=True))

# Optional: just the list of the 20 drug names
top20_names_orig = top20['pert_iname'].tolist()



# ensure numeric
final_df_pred['reverse_score'] = pd.to_numeric(final_df_pred['reverse_score'], errors='coerce')

# 1) pick the row with the max reverse_score within each pert_iname
best_idx = final_df_pred.groupby('pert_iname')['reverse_score'].idxmax()
top_per_drug = final_df_pred.loc[best_idx]

# 2) sort those by reverse_score and take the top 20
top20 = (top_per_drug
         .sort_values('reverse_score', ascending=False)
         .head(20)
         .reset_index(drop=True))

# Optional: just the list of the 20 drug names
top20_names_pred = top20['pert_iname'].tolist()

'''
how many overlapped
'''
overlap = set(top20_names_orig) & set(top20_names_pred)
n_overlap = len(overlap)          # 5
# optional:
overlap_list = sorted(overlap)    # ['GSK-2126458', 'NVP-BEZ235', 'PI-103', 'ZSTK-474', 'torin-2']

'''
how many drugs in the inference set is not in the original set.
'''
drug_list_orig = final_df_orig['pert_iname'].unique().tolist()

orig_set = set(drug_list_orig)
missing_from_orig = [x for x in top20_names_pred if x not in orig_set] # 'WAY-262611', 'BG-1010'
n_missing = len(missing_from_orig)   # 2


'''
top50 drugs
'''
# 1) pick the row with the max reverse_score within each pert_iname
best_idx = final_df_orig.groupby('pert_iname')['reverse_score'].idxmax()
top_per_drug = final_df_orig.loc[best_idx]

# 2) sort those by reverse_score and take the top 20
top50 = (top_per_drug
         .sort_values('reverse_score', ascending=False)
         .head(50)
         .reset_index(drop=True))

# Optional: just the list of the 20 drug names
top50_names_orig = top50['pert_iname'].tolist()



# ensure numeric
final_df_pred['reverse_score'] = pd.to_numeric(final_df_pred['reverse_score'], errors='coerce')

# 1) pick the row with the max reverse_score within each pert_iname
best_idx = final_df_pred.groupby('pert_iname')['reverse_score'].idxmax()
top_per_drug = final_df_pred.loc[best_idx]

# 2) sort those by reverse_score and take the top 20
top50 = (top_per_drug
         .sort_values('reverse_score', ascending=False)
         .head(50)
         .reset_index(drop=True))

# Optional: just the list of the 20 drug names
top50_names_pred = top50['pert_iname'].tolist()

'''
how many overlapped
'''
overlap = set(top50_names_orig) & set(top50_names_pred)
n_overlap = len(overlap)          # 16
# optional:
overlap_list = sorted(overlap)    # ['AZD-8055', 'CD-437', 'GSK-1059615', 'GSK-2126458', 'JW-7-24-1', 'KU-0060648', 'NVP-BEZ235', 'NVP-TAE684', 'PD-0325901', 'PD-184352', 'PI-103', 'WYE-125132', 'ZSTK-474', 'foretinib', 'torin-1', 'torin-2']

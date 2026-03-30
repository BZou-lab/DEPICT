'''
Downloaded from:
https://aacrjournals.org/mct/article/15/6/1155/92159/An-Unbiased-Oncology-Compound-Screen-to-Identify
Supplementary data
'''
import pandas as pd
'''
the drug synergy reference dataset
'''
df_comb = pd.read_excel("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/15357163mct150843-sup-156849_1_supp_1_w2lrww.xls")
cell_counts = df_comb['cell_line'].value_counts()
df_single = pd.read_excel("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/15357163mct150843-sup-156849_1_supp_0_w2lh45.xlsx")


'''
the lincs dataset
'''
import scanpy as sc
adata = sc.read('./Data/FinalData/adataAfterClean.h5ad')
sc.pp.normalize_total(adata)

adata_ht29 = adata[adata.obs['cell_id']=='HT29'].copy()

adata_ht29.write("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/ht29.h5ad")



'''
find common cells
'''
adata.obs['cell_id'].nunique() # 82
df_comb['cell_line'].nunique() # 39

cells_adata = adata.obs['cell_id'].str.lower().unique()

# From DataFrame
cells_df = df_comb['cell_line'].str.lower().unique()

# Find common
common_cells = set(cells_adata).intersection(set(cells_df))

print(f"Number of unique common cells: {len(common_cells)}") # 8 common cells
print("Examples:", list(common_cells)) # 'rko', 'hct116', 'lncap', 'ht29', 'lovo', 'sw620', 'vcap', 'a375'

'''
find common drugs for HT29
'''
HT29 = df_comb[df_comb['cell_line']=='HT29'].copy()
HT29_single = df_single[df_single['cell_line']=='HT29']

drugs_adata = adata.obs['pert_iname'].str.lower().unique()
drugs_ht29 = pd.unique(
    pd.concat([HT29['drugA_name'], HT29['drugB_name']])
    .str.lower()
)
drugs_ht29_single = HT29_single['drug_name'].str.lower().unique()
common_drugs = set(drugs_adata).intersection(set(drugs_ht29))
common_drugs_ht29 = set(drugs_ht29_single).intersection(set(drugs_ht29))

print(f"Number of unique common cells: {len(common_drugs)}") # 22 common drugs for HT29.
print("Examples:", list(common_drugs))
'''
common drugs:
['paclitaxel', 'topotecan', 'sunitinib', 'lapatinib', 'methotrexate', 'etoposide', 
'metformin', 'temozolomide', 'doxorubicin', 'cyclophosphamide', 'mk-2206', 'sn-38', 
'vinorelbine', 'geldanamycin', 'mk-5108', 'dexamethasone', 'sorafenib', 'vinblastine', 
'dasatinib', 'erlotinib', 'bortezomib', 'gemcitabine']
'''
common_drugs = {d.lower() for d in common_drugs}

# make safe lowercase views (handles NaNs)
a = HT29['drugA_name'].astype(str).str.lower()
b = HT29['drugB_name'].astype(str).str.lower()

# keep rows where BOTH drugs are in common_drugs
mask = a.isin(common_drugs) & b.isin(common_drugs)

HT29_filtered = HT29[mask].copy()

c = HT29_single['drug_name'].astype(str).str.lower()
mask1 = c.isin(common_drugs)
HT29_single_filtered = HT29_single[mask1].copy()

HT29_single_filtered.nunique()
HT29_single_filtered.columns
'''
next steps:
1.  find the duration of the reference experiments in the original paper;
    figure out what is the viability1 to 4; what is mu/muMax; what is X/X0
    The study used 4 by 4 dosing regime, which means each drug only has 4 unique dosages,
    combining with other drugs of 4 unique dosages which is 16 total dosages regime for one combination of drugs.
Answer to 1: 96 hours proliferation.    

2.  prepare the screening dataset: for only the 22 common drugs for the used dosage and duration.

3.  get the prediction from the LINCS model.

4.  build a binary classifier for those 2,688 experiments to see if the can do a good classfication based on 
    our prediction using the LINCS framework.
'''

'''
step 1: figure out the data
first check out the range 
'''
max(HT29_filtered['viability1']) # 1.27328
min(HT29_filtered['viability1']) # 0.00105

max(HT29_filtered['viability2']) # 1.38219
min(HT29_filtered['viability2']) # 0.00048

max(HT29_filtered['viability3']) # 1.27952
min(HT29_filtered['viability3']) # 0.00051

max(HT29_filtered['viability4']) # 1.37542
min(HT29_filtered['viability4']) # 0.00064

max(HT29_filtered['mu/muMax']) # 1.09077
min(HT29_filtered['mu/muMax']) # -1.92618

max(HT29_filtered['X/X0']) # 1.13409
min(HT29_filtered['X/X0']) # 0.01731

max(HT29_single_filtered['X/X0']) # 1.09216
min(HT29_single_filtered['X/X0']) # 0.05531

HT29_filtered.to_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_pairData.csv")
HT29_single_filtered.to_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_singleData.csv")

HT29_filtered['combination_name'].nunique() # 165, why?
'''
there are 2688 points, so should be 2688/16=168 drug combinations but only 165 drug combinations, why?
'''
drug_pair_count = HT29_filtered['combination_name'].value_counts()

MK2206_1 = HT29_filtered[HT29_filtered['combination_name']=='Sunitinib & MK-2206']
MK2206_2 = HT29_filtered[HT29_filtered['combination_name']=='Lapatinib & MK-2206']
MK2206_3 = HT29_filtered[HT29_filtered['combination_name']=='MK-2206 & Erlotinib']
'''
for the above three combinations, there are two batches, so for every dosage regime, there are two replicates.
'''
'''
check the same thing for single-agent
'''
drug_pair_count_single = HT29_single_filtered['drug_name'].value_counts()
'''
same issue
'''

'''
check the batch
1. for drug synergy dataset:
    2640 from batch 1 but only 48 from batch 3, and also the 48 samples have the same replicate in the batch 1
    So keep only batch 1 as hard to deal with batch effect.

2. for single-agent dataset:
    drop batch 3 too, because drugs in the batch 3 are not common in the synergy dataset.
'''
batch_counts = HT29_filtered['BatchID'].value_counts()
batch_counts_single = HT29_filtered['BatchID'].value_counts()

HT29_filtered2 = HT29_filtered[HT29_filtered['BatchID']==1].copy()
HT29_single_filtered2 = HT29_single_filtered[HT29_single_filtered['BatchID']==1].copy()
'''
sanity check
'''
# check unique drugs or drug combination
HT29_filtered2['combination_name'].nunique() # 165, which is 2640/16
HT29_single_filtered2['drug_name'].nunique() # 22, which is 176/8
# check common drugs
drugs_ht29 = pd.unique(
    pd.concat([HT29_filtered2['drugA_name'], HT29_filtered2['drugB_name']])
    .str.lower()
)
drugs_ht29_single = HT29_single_filtered2['drug_name'].str.lower().unique()
common_drugs_ht29 = set(drugs_ht29_single).intersection(set(drugs_ht29))
len(common_drugs_ht29) # 22, good.
'''
final df output
'''
HT29_filtered2.to_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_pairData.csv")
HT29_single_filtered2.to_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_singleData.csv")

'''
check the read-in data
'''
HT29_filtered2_read = pd.read_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_pairData.csv", index_col=0)
HT29_single_filtered2_read = pd.read_csv("./Code/downstream_analysis_code/DrugSynergyPrediction/Data/HT29_singleData.csv", index_col=0)

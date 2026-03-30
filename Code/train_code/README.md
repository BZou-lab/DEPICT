# train_code
The two python files are the best performed codes among many hyperparameter grids. 

The two codes are the same, but one is for initial training and another for continuing training from the initial training.

## Transformer_d32h8l4_dp1_MSECor_lr1CosSche_3XAttn_sepGeneEncPred_newAtten_FiLM_Enc3DimReducLa128_CellAware_frontEndMLPSimp_DoseTimeAsScalarXattns_LLMMFPonly_first50Epoch.py
Please run this code first, this is the initial training code. This code will only run 50 epochs.

### Example use
"python Transformer_d32h8l4_dp1_MSECor_lr1CosSche_3XAttn_sepGeneEncPred_newAtten_FiLM_Enc3DimReducLa128_CellAware_frontEndMLPSimp_DoseTimeAsScalarXattns_LLMMFPonly_first50Epoch.py --split_type cell_split1"

Argument: 

'split_type', the split strategy, see paper for more details for each split strategy. choices=['random_split1', 'cell_split1', 'drug_split1','random_split2', 'cell_split2', 'drug_split2', 'random_split3', 'cell_split3', 'drug_split3', 'random_split4', 'cell_split4', 'drug_split4', 'random_split5', 'cell_split5', 'drug_split5']. 

Each split strategy has 5 different repetitions, aiming for generate a robust benchmarking performance comparison among different predictive frameworks.


## Transformer_d32h8l4_dp1_MSECor_lr1CosSche_3XAttn_sepGeneEncPred_newAtten_FiLM_Enc3DimReducLa128_CellAware_frontEndMLPSimp_DoseTimeAsScalarXattns_LLMMFPonly_rest150Epoch.py
Run this if training 50 epochs is not enough for you.

### Example use
"python Transformer_d32h8l4_dp1_MSECor_lr1CosSche_3XAttn_sepGeneEncPred_newAtten_FiLM_Enc3DimReducLa128_CellAware_frontEndMLPSimp_DoseTimeAsScalarXattns_LLMMFPonly_rest150Epoch.py --split_type drug_split1 --num_epochs 58 --start_epoch 93"

Argument: 

'split_type': the split strategy, see paper for more details for each split strategy. choices=['random_split1', 'cell_split1', 'drug_split1','random_split2', 'cell_split2', 'drug_split2', 'random_split3', 'cell_split3', 'drug_split3', 'random_split4', 'cell_split4', 'drug_split4', 'random_split5', 'cell_split5', 'drug_split5']. 

'num_epochs': number of epochs will be run in this code.

'start_epoch': the starting epochs, depending on the log from the last training, or which start epoch you want.

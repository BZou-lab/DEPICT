# inference_code

## prediction.py
Run this after you trained the models of interest.

## Important note
change the 'ckpt_name' in the code correspondingly if you are using different naming.
default in the code is: ckpt_name = f"transformer_d32h8l4_{args.split_type}_dp1_MSECor_lr1_CosSche_3XAttn_sepGene2EncPred_newAttn_FiLM_Enc3DimReducLa128_CellAware_frontEndMLPsimp_whole_DoseTimeAsScalarXattns_LLMMFP_first50epoch.pth"


### Example use
"python prediction.py --split_type drug_split3 --subset test --action statistics"

Arguments:

'split_type': the split strategy, see paper for more details for each split strategy. choices=['random_split1', 'cell_split1', 'drug_split1','random_split2', 'cell_split2', 'drug_split2', 'random_split3', 'cell_split3', 'drug_split3', 'random_split4', 'cell_split4', 'drug_split4', 'random_split5', 'cell_split5', 'drug_split5']. 

'subset': the subset that will be inferred by the trained model. choices=["validation", "test"]

'action' the action that will be taken from the inference. It can be prediction or just calculating the statistics for the model or both. choices=["prediction", "statistics", "both"]
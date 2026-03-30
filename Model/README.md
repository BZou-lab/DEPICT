# Trained models and their loss track during training.
This folder contains all the trained DEPICT models and the training loss during the training process.

## .pth files
The naming system is: 

transformer_d32h8l4_{split_type}_dp1_MSECor_lr1_CosSche_3XAttn_sepGene2EncPred_newAttn_FiLM_Enc3DimReducLa128_CellAware_frontEndMLPsimp_whole_DoseTimeAsScalarXattns_LLMMFP_first50epoch.pth.

Where split_type: choices=['random_split1', 'cell_split1', 'drug_split1','random_split2', 'cell_split2', 'drug_split2', 'random_split3', 'cell_split3', 'drug_split3', 'random_split4', 'cell_split4', 'drug_split4', 'random_split5', 'cell_split5', 'drug_split5']

Thus, there are 15 final models.

## TrainingLoss
This folder contains the training losses during training for the above 15 final models.
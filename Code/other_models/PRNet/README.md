# PRNet
This folder contains codes and data that train and infer for PRNet, used in my study. The original GitHub page for PRNet is: https://github.com/Perturbation-Response-Prediction/PRnet. 

The three modified python files are 'train_lincs_final_run.py', 'test_lincs_final_run_first50ep.py' and 'test_lincs_final_run_rest50ep.py'. 

## Important 
Please clone the whole git repository from the original PRNet study before runing my modified code. The original repository contains many must-have codes to run my modified version.

## Data
PRNew will be using the same data file as our study. which is the './Data/FinalData/adataAfterClean.h5ad'

## train_lincs_final_run.py
This code was used to train PRNet.
### Example use
"python train_lincs_final_run.py --split_key drug_split1"

For resuming training from initial training if 50 epochse is not enough.
"python train_lincs_final_run.py --split_key drug_split5 --resume --extra_epochs 50"

'split_key': the split strategy, see paper for more details for each split strategy. choices=['random_split1', 'cell_split1', 'drug_split1','random_split2', 'cell_split2', 'drug_split2', 'random_split3', 'cell_split3', 'drug_split3', 'random_split4', 'cell_split4', 'drug_split4', 'random_split5', 'cell_split5', 'drug_split5']. 

'resume': Controls this run as a resuming training or an initial training. If nothing in the command, then initial training, if defined '--resume', then contining training based on an existing model.

'extra_epochs': how many epochs should be run for the resuming training.


## test_lincs_final_run_first50ep.py
This code was used after 50 epochs training of PRNet.
### Example use
"python test_lincs_final_run_first50ep.py --split_key cell_split2"

'split_key': the split strategy, see paper for more details for each split strategy. choices=['random_split1', 'cell_split1', 'drug_split1','random_split2', 'cell_split2', 'drug_split2', 'random_split3', 'cell_split3', 'drug_split3', 'random_split4', 'cell_split4', 'drug_split4', 'random_split5', 'cell_split5', 'drug_split5']. 

## test_lincs_final_run_rest50ep.py
This code was used after 100 epochs training of PRNet.
### Example use
"python test_lincs_final_run_rest50ep.py --split_key random_split5"

'split_key': the split strategy, see paper for more details for each split strategy. choices=['random_split1', 'cell_split1', 'drug_split1','random_split2', 'cell_split2', 'drug_split2', 'random_split3', 'cell_split3', 'drug_split3', 'random_split4', 'cell_split4', 'drug_split4', 'random_split5', 'cell_split5', 'drug_split5']. 

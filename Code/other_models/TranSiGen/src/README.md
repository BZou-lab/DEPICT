# src
This folder contains all the codes that train and infer for TranSiGen, used in my study. The original GitHub page for PRNet is: https://github.com/myzhengSIMM/TranSiGen. 

Please see codes for detailed arguments for each python file.

## train_TranSiGen_full_data_Meisheng.py
This code is used to train the TranSiGen by the same datasets used in DEPICT and PRNet.

### Example use
"python train_TranSiGen_full_data_Meisheng.py --data_path ../data/Meisheng_used_data/processed_data_id.h5 --molecule_path ../data/Meisheng_used_data/idx2smi.pickle --molecule_feature ECFP4 --initialization_model pretrain_shRNA --split_data_type drug_split1 --n_epochs 300 --n_latent 100 --molecule_feature_embed_dim 400 --batch_size 128 --learning_rate 1e-3 --beta 0.1 --dropout 0.1 --weight_decay 1e-5 --train_flag True --eval_metric True"

These arguments: "--n_latent 100 --molecule_feature_embed_dim 400 --batch_size 128 --learning_rate 1e-3 --beta 0.1 --dropout 0.1 --weight_decay 1e-5" are default setting from TranSiGen.

## prediction_Meisheng.py
This is to get the prediction from the trained models.

### Example use
"python prediction_Meisheng.py --model_path ../results/trained_models_drug_split1_epoch_300/feature_ECFP4_init_pretrain_shRNA/best_model.pt --molecule_feature ECFP4 --split_data_type drug_split1 --subset test --n_epoch 300"

## prediction_Meisheng_getStats.py
This is to get the performance metrics only from the trained models.

### Example use
"python prediction_Meisheng_getStats.py --model_path ../results/trained_models_cell_split1_epoch_300/feature_ECFP4_init_pretrain_shRNA/best_model.pt --molecule_feature ECFP4 --split_data_type cell_split1 --subset test --n_epoch 300"

# -*- coding: utf-8 -*-
# @Author: Xiaoning Qi
# @Date:   2022-06-13 09:47:44
# @Last Modified by:   Xiaoning Qi
# @Last Modified time: 2024-11-04 15:56:30
# @Last Modified by:   Meisheng Xiao
# @Last Modified time: Jul19 2025 (whole dataset); Feb22 2025(small dataset)
# use a small subset and epoch is 50, also mute the scale (line 69)
# small subset (line 67), epoch (line 40)
# @Last Modified time: Aug1 2025
# @ added some parsers to make the wrapper can do continue training.

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import sys
print(sys.path)

import argparse 
from datetime import datetime
import scanpy as sc
from trainer.PRnetTrainer import PRnetTrainer
import torch

def parse_args():
    parse = argparse.ArgumentParser(description='perturbation-conditioned generative model')  
    parse.add_argument('--split_key', default='random_split_0', type=str, help='split key of data') 
    parse.add_argument('--resume', action='store_true',
                       help='continue training from save_dir/resume.pt')
    parse.add_argument('--extra_epochs', type=int, default=0,
                       help='extra epochs to run when --resume is given')
    args = parse.parse_args()  
    return args



if __name__ == "__main__":
    args_train = parse_args()
    start_time = datetime.now()



    config_kwargs = {
        'batch_size' : 512,
        'comb_num' : 1,
        'save_dir' : './checkpoint/final_run/',
        'n_epochs' : 50,
        'split_key' : args_train.split_key,
        'x_dimension' : 978,
        'hidden_layer_sizes' : [128],
        'z_dimension' : 64,
        'adaptor_layer_sizes' : [128],
        'comb_dimension' : 64, 
        #'drug_dimension': 1031,
        'drug_dimension': 1024,
        'dr_rate' : 0.05,
        'lr' : 1e-3, 
        'weight_decay' : 1e-8,
        'scheduler_factor' : 0.5,
        'scheduler_patience' : 10,
        'n_genes' : 20,
        'loss' : ['GUSS'], 
        'obs_key' : 'cov_drug_name'
    }  


    


    print(os.getcwd())

    # adata = sc.read('./dataset/Lincs_L1000.h5ad')
    # read the subset by Meisheng
    # adata = sc.read('./dataset/Lincs_L1000_small.h5ad')
    # read the subset by Meisheng
    adata = sc.read('./Data/FinalData/adataAfterClean.h5ad')
    sc.pp.normalize_total(adata)
    # sc.pp.log1p(adata)

    Trainer = PRnetTrainer(
                            adata,
                            batch_size=config_kwargs['batch_size'],
                            comb_num=config_kwargs['comb_num'],
                            split_key=config_kwargs['split_key'],
                            model_save_dir=config_kwargs['save_dir'],
                            x_dimension=config_kwargs['x_dimension'],
                            hidden_layer_sizes=config_kwargs['hidden_layer_sizes'],
                            z_dimension=config_kwargs['z_dimension'],
                            adaptor_layer_sizes=config_kwargs['adaptor_layer_sizes'],
                            comb_dimension=config_kwargs['comb_dimension'],
                            drug_dimension=config_kwargs['drug_dimension'],
                            dr_rate=config_kwargs['dr_rate'],
                            n_genes=config_kwargs['n_genes'],
                            loss = config_kwargs['loss'],
                            obs_key = config_kwargs['obs_key']
                                )
    
    # ── NEW: handle resume option ────────────────────────────────
    os.makedirs(config_kwargs['save_dir'], exist_ok=True)

    if args_train.resume:
        ckpt_path = os.path.join(
            config_kwargs['save_dir'],
            f"{config_kwargs['split_key']}_resume.pt"    # ← must match your save name
        )
        assert os.path.isfile(ckpt_path), f"No checkpoint found at {ckpt_path}"

        # 1) Build optimizer & scheduler so we have targets to load state into
        Trainer.build_optim(
            lr=config_kwargs['lr'],
            weight_decay=config_kwargs['weight_decay'],
            scheduler_factor=config_kwargs['scheduler_factor'],
            scheduler_patience=config_kwargs['scheduler_patience'],
        )

        # 2) Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=Trainer.device)

        # 3) Robustly load model weights (strip 'module.' if resuming on 1 GPU)
        state = ckpt['model_state']
        if list(state.keys())[0].startswith('module.') and not isinstance(Trainer.modelPGM, torch.nn.DataParallel):
            state = {k.replace('module.', '', 1): v for k, v in state.items()}
        Trainer.modelPGM.load_state_dict(state, strict=True)

        # 4) Load optimizer/scheduler & trainer bookkeeping
        Trainer.optimPGM.load_state_dict(ckpt['optim_state'])
        Trainer.scheduler_autoencoder.load_state_dict(ckpt['sched_state'])
        Trainer.best_mse = ckpt.get('best_mse', float('inf'))
        Trainer.patient  = ckpt.get('patient', 0)
        Trainer.epoch    = ckpt.get('epoch', -1)

        # 5) How many more epochs to run now
        epochs_this_run = args_train.extra_epochs
        if epochs_this_run <= 0:
            raise ValueError("--extra_epochs must be > 0 when --resume is used")
    else:
        epochs_this_run = config_kwargs['n_epochs']
# ─────────────────────────────────────────────────────────────


    #Trainer.train(
    #    n_epochs = config_kwargs['n_epochs'],
    #    lr = config_kwargs['lr'], 
    #    weight_decay= config_kwargs['weight_decay'], 
    #    scheduler_factor=config_kwargs['scheduler_factor'],
    #    scheduler_patience=config_kwargs['scheduler_patience'])
    
    Trainer.train(
        n_epochs=epochs_this_run,
        lr=config_kwargs['lr'],
        weight_decay=config_kwargs['weight_decay'],
        scheduler_factor=config_kwargs['scheduler_factor'],
        scheduler_patience=config_kwargs['scheduler_patience']
    )

    end_time = datetime.now()

    during_time = (end_time-start_time).seconds/60

    print(f'start time: {start_time} end_time: {end_time} time:{during_time} min')

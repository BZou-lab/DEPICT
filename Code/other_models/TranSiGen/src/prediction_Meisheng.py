'''
Modified by Meisheng.
This prediction only gives the 978 landmark genes on valid/test set
1. deleted items in argparser: seed, cell; added the split type
2. deleted the whole genome genes inference.

'''
from dataset import TranSiGenDataset
from utils import *
from cmapPy.pandasGEXpress.parse import parse
import pickle
import argparse
import warnings
import torch
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for prediction")
    parser.add_argument("--model_path", type=str, default='../results/trained_models_164_cell_smiles_split/364039/feature_KPGT_init_pretrain_shRNA/best_model.pt')
    parser.add_argument("--data_path", type=str, default='./Code/other_models/TranSiGen/data/Meisheng_used_data/processed_data_id.h5')
    parser.add_argument("--molecule_feature", type=str, default='ECFP4', help='molecule_feature(KPGT, ECFP4)')
    parser.add_argument("--split_data_type", type=str, default='random_split1', help='split_data_type(random_split1, cell_split1, drug_split1), etc')
    parser.add_argument("--subset", choices=["validation", "test"], default="validation")
    parser.add_argument("--dev", type=str, default='cuda:0')
    parser.add_argument("--n_epoch", type=str, default='100', help='indication for the model trained how many epochs')

    args = parser.parse_args()
    return args

def prediction_profiles(args):
    # df_gene = pd.read_csv('../data/LINCS2020/geneinfo_processed.csv')
    # df_landmark_gene = df_gene[(df_gene['pr_is_bing'] == 1) & (df_gene['pr_is_lm']==1)]
    # df_best_infer_gene = df_gene[(df_gene['pr_is_bing'] == 1) & (df_gene['pr_is_lm']==0)]
    # landmark_ids = df_landmark_gene['pr_id'].tolist()
    # best_infer_ids = df_best_infer_gene['pr_id'].tolist()
    # weight_path = '../data/LINCS2020/infer_weight.gctx'
    # infer_weight = parse(weight_path, cid=['OFFSET']+landmark_ids, rid=best_infer_ids)
    # infer_weight_df_tmp = infer_weight.data_df
    # infer_weight_df = infer_weight_df_tmp[['OFFSET'] + landmark_ids]
    # infer_weight_df = infer_weight_df.loc[best_infer_ids]

    n_epoch = args.n_epoch
    split_type = args.split_data_type
    subset_tag = "validation" if (args.subset == "validation") else "test"
    feat_type = args.molecule_feature
    if feat_type == 'KPGT':
        features_dim = 2304
    elif feat_type == 'ECFP4':
        features_dim = 2048
    dev = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model_path, map_location='cpu')
    model.dev = torch.device(dev)
    model.to(dev)
    print(model)

    # selected_cid = args.cell
    # random_seed = args.seed
    # df_screening = pd.read_csv(args.data_path)

    # emb_array = []
    # smi_idx_array = []
    # for idx, row in df_screening.iterrows():
    #     smi = row['canonical_smiles']
    #     emb_array.append(smi2emb[smi])
    #     smi_idx_array.append(row['cp_id'])
    # emb_array = np.array(emb_array)
    # smi_idx_array = np.array(smi_idx_array)
    # cid_array = np.array([selected_cid] * emb_array.shape[0])
    # x1_array = dict_modz_x1_all_cid[selected_cid]
    # x1_array = np.repeat(x1_array, emb_array.shape[0], axis=0).astype(np.float32)

    # test = TranSiGenDataset_screening(x1=x1_array, mol_feature=emb_array, mol_id=smi_idx_array, cid=cid_array)
    # test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=64, shuffle=False, drop_last=False, num_workers=4, worker_init_fn=seed_worker)
    data = load_from_HDF(args.data_path)
        
    pair, pairv, pairt = split_data_meisheng(data, split_key=split_type, verbose=False)
    print('===============', split_type, '================')
    print('train', len(set(pair['cid'])), len(pair['canonical_smiles']), len(set(pair['canonical_smiles'])), )
    print('valid', len(set(pairv['cid'])), len(pairv['canonical_smiles']), len(set(pairv['canonical_smiles'])), )
    print('test', len(set(pairt['cid'])), len(pairt['canonical_smiles']), len(set(pairt['canonical_smiles'])), )
    
    train = TranSiGenDataset(
        LINCS_index=pair['LINCS_index'],
        mol_feature_type=feat_type,
        mol_id=pair['canonical_smiles'],
        cid=pair['cid']
    )

    valid = TranSiGenDataset(
        LINCS_index=pairv['LINCS_index'],
        mol_feature_type=feat_type,
        mol_id=pairv['canonical_smiles'],
        cid=pairv['cid']
    )

    test = TranSiGenDataset(
        LINCS_index=pairt['LINCS_index'],
        mol_feature_type=feat_type,
        mol_id=pairt['canonical_smiles'],
        cid=pairt['cid']
    )


    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=64, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)
    valid_loader = torch.utils.data.DataLoader(dataset=valid, batch_size=64, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=64, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)

    # setup_seed(random_seed)
    validation_pred = (args.subset == "validation")
    
    if validation_pred:
        print('predicting the validation set')
        for name, loader in zip(['test'], [valid_loader]):
            x1_array, x2_array, x1_rec_array, x2_rec_array, x2_pred_array, _, mol_id_array, cid_array, sig_array = model.predict_profile(loader=loader)
            ddict_data = dict()
            ddict_data['x1'] = x1_array
            ddict_data['x2'] = x2_array
            ddict_data['x2_rec'] = x2_rec_array
            ddict_data['x2_pred'] = x2_pred_array
            ddict_data['cp_id'] = mol_id_array
            ddict_data['cid'] = cid_array
            ddict_data['sig'] = sig_array
    else:
        print('predicting the test set')
        for name, loader in zip(['validation'], [test_loader]):
            x1_array, x2_array, x1_rec_array, x2_rec_array, x2_pred_array, _, mol_id_array, cid_array, sig_array = model.predict_profile(loader=loader)
            ddict_data = dict()
            ddict_data['x1'] = x1_array
            ddict_data['x2'] = x2_array
            ddict_data['x2_rec'] = x2_rec_array
            ddict_data['x2_pred'] = x2_pred_array
            ddict_data['cp_id'] = mol_id_array
            ddict_data['cid'] = cid_array
            ddict_data['sig'] = sig_array

    for k in ddict_data.keys():
        print(k, ddict_data[k].shape)
        
    
    
    save_to_HDF('./Code/other_models/TranSiGen/results/LMGenes_Prediction/prediction_profile_{}_{}_{}.h5'.format(split_type,subset_tag,n_epoch), ddict_data)




if __name__ == "__main__":
    args = parse_args()
    prediction_profiles(args)

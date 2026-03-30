'''
Sep 1, 2025.
Meisheng Xiao
This python file is used to get the prediction of the validation or test set.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from torch.utils.data import Dataset, DataLoader
import math
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import time as pytime
import argparse
import os
from scipy.stats import pearsonr
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Argument parser for command-line execution
parser = argparse.ArgumentParser(description="Train model on gene expression data with specified split type.")
parser.add_argument('--split_type', type=str, required=True, choices=['random_split1', 'cell_split1', 'drug_split1',
                                                                      'random_split2', 'cell_split2', 'drug_split2',
                                                                      'random_split3', 'cell_split3', 'drug_split3',
                                                                      'random_split4', 'cell_split4', 'drug_split4',
                                                                      'random_split5', 'cell_split5', 'drug_split5'],
                    help="Specify the type of data split. Total of 5 different dataset from different spliting for Meisheng used dataset")
parser.add_argument("--subset", choices=["validation", "test"], default="validation", help = "choose the action over validation or test set")
parser.add_argument("--action", choices=["prediction", "statistics", "both"], default="statistics", help="what action will be done on this code; Prediction will give the prediction on the test/validation set; statistics will only calculate the statistics; both will do both action")
args = parser.parse_args()

adata = sc.read('./Data/FinalData/adataAfterClean.h5ad')
gptEmbed_df = pd.read_csv('./Data/FinalData/gptEmbed_Jul9_final.csv',index_col=0)
MFP_df = pd.read_csv("./Data/FinalData/compounds_512MFP_wholeDat_fixed.csv",index_col=0)
drug_targets = pd.read_csv("./Data/FinalData/compounds_target_multihot_full.csv", index_col=0)
sc.pp.normalize_total(adata)

subset_tag = "validation" if (args.subset == "validation") else "test"
validation_pred = (args.subset == "validation")
act_pred = (args.action == "prediction")
act_stat = (args.action == "statistics")
act_both = (args.action == "both")

class GenePerturbationDataset(Dataset):
    """Returns tensors for baseline, drug features, *cell stats* and target.

    Adds a 1 956‑dim `cell_stats` (= mean ‖ variance per gene) **without touching
    the rest of your original logic**.
    """

    def __init__(self, adata, gptEmbed_df, MFP_df, drug_targets,
                 split_strategy='cell_split', split_value='train'):
        self.adata        = adata
        self.gptEmbed_df  = gptEmbed_df
        self.MFP_df       = MFP_df
        self.drug_targets = drug_targets
        self.indices      = np.where(adata.obs[split_strategy] == split_value)[0]

        # pre‑compute per‑cell mean & variance *across baseline controls*
        ctrl_mask      = adata.obs['control'] == 1
        X_ctrl         = adata.X[ctrl_mask]
        X_ctrl         = X_ctrl.toarray() if hasattr(X_ctrl, 'toarray') else X_ctrl
        cell_ids_ctrl  = adata.obs.loc[ctrl_mask, 'cell_id'].values

        self.cell_stats = {}
        for cid in np.unique(cell_ids_ctrl):
            mat = X_ctrl[cell_ids_ctrl == cid]
            mu  = mat.mean(0, dtype=np.float32)
            var = mat.var(0, dtype=np.float32)
            self.cell_stats[cid] = torch.from_numpy(np.concatenate([mu, var]))  # (1956,)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx   = self.indices[idx]
        y_true     = self.adata.X[real_idx]

        # paired control → baseline expression
        paired_id  = self.adata.obs.iloc[real_idx]['paired_control_index']
        paired_idx = self.adata.obs.index.get_loc(paired_id)
        x_baseline = self.adata.X[paired_idx]

        # cell‑level stats lookup via cell_id of the *control* row
        cell_id    = self.adata.obs.iloc[paired_idx]['cell_id']
        cell_stats = self.cell_stats[cell_id]                       # (1956,)

        # drug features ------------------------------------------------
        drug_name       = self.adata.obs.iloc[real_idx]['pert_iname']
        x_drug_gptEmbed = self.gptEmbed_df.loc[drug_name].values.astype('float32')
        x_drug_MFP      = self.MFP_df.loc[drug_name].values.astype('float32')
        x_drug_targets  = self.drug_targets.loc[drug_name].values.astype('float32')
        
        # dosage and duration
        dose = float(self.adata.obs.iloc[real_idx]['dose'])       # µM
        time = float(self.adata.obs.iloc[real_idx]['pert_time'])  # h

        dose_feat = torch.tensor(math.log10(dose + 1.0), dtype=torch.float32)
        time_feat = torch.tensor(math.log10(time + 1.0), dtype=torch.float32)

        # to torch tensors ---------------------------------------------
        return (
            torch.tensor(x_baseline,       dtype=torch.float32),  # (978,)
            cell_stats,                                         # (1956,)
            torch.tensor(x_drug_gptEmbed, dtype=torch.float32),  # (512,)
            torch.tensor(x_drug_MFP,      dtype=torch.float32),  # (512,)
            torch.tensor(x_drug_targets,  dtype=torch.float32),  # (1164,)
            torch.tensor(y_true,          dtype=torch.float32),   # (978,)
            dose_feat, # scalar
            time_feat # scalar
        )


# Create datasets for different splits
split_type = args.split_type
train_dataset = GenePerturbationDataset(adata, gptEmbed_df, MFP_df, drug_targets, split_type, "train")
valid_dataset = GenePerturbationDataset(adata, gptEmbed_df, MFP_df, drug_targets, split_type, "valid")
test_dataset = GenePerturbationDataset(adata, gptEmbed_df, MFP_df, drug_targets, split_type, "test")


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ────────────────────────────────────────────────────────────────────────────
# Utility: lightweight denoising auto‑encoder
# ────────────────────────────────────────────────────────────────────────────
class DenoisingAE(nn.Module):
    """Two-layer symmetric auto-encoder with configurable bottleneck."""
    def __init__(self, in_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        h_dim = max(latent_dim * 2, in_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h_dim, latent_dim), nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h_dim, in_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


# ╭─────────────────────────────────────────────────────────────╮
# │ 1.  Encoder‑only projectors                                 │
# ╰─────────────────────────────────────────────────────────────╯
class Enc2Layer(nn.Module):
    """
    Two‑layer MLP encoder (same shape as the old AE encoder).
    512 → h_dim → latent_dim
    """
    def __init__(self, in_dim: int = 512,
                       latent_dim: int = 128,
                       dropout: float = 0.1):
        super().__init__()
        h_dim = max(latent_dim * 2, in_dim // 2)      # mirror AE logic
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, latent_dim),
            nn.GELU()
        )

    def forward(self, x):           # (B, in_dim) → (B, latent_dim)
        return self.net(x)


class Enc3Layer(nn.Module):
    """
    Three‑layer funnel: 512 → 256 → 128 → latent_dim.
    BatchNorm stabilises deeper stack; second dropout is lighter.
    """
    def __init__(self, in_dim: int = 512,
                       latent_dim: int = 128,
                       dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),    nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, latent_dim)                 # no activation = linear code
        )

    def forward(self, x):           # (B, in_dim) → (B, latent_dim)
        return self.net(x)


'''
Model Transformer encoder for gene
'''
class GenePerturbationTransformer(nn.Module):
    def __init__(self,
                 d_model: int = 32,
                 num_heads: int = 8,
                 num_encoder_layers: int = 2,
                 dropout: float = 0.2,
                 max_len: int = 978,
                 d_hidden: int = 64,
                 n_genes_target: int = 1164,
                 fp_bits: int = 512,
                 llm_dim: int = 512,
                 ae_latent: int = 128):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.max_len   = max_len

        # ───── gene‑specific **two‑layer** MLP  (3 → d_mid → d_model)  ----------
        d_mid = d_model // 2  # e.g. 16 when d_model=32
        # first layer weights/bias  (S, 3, d_mid)
        self.W1_gene = nn.Parameter(torch.empty(max_len, 3, d_mid))
        self.b1_gene = nn.Parameter(torch.empty(max_len, d_mid))
        # second layer  (S, d_mid, d_model)
        self.W2_gene = nn.Parameter(torch.empty(max_len, d_mid, d_model))
        self.b2_gene = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.xavier_uniform_(self.W1_gene)
        nn.init.zeros_(self.b1_gene)
        nn.init.xavier_uniform_(self.W2_gene)
        nn.init.zeros_(self.b2_gene)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.gene_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # ───────────── drug-feature projections & fusion  ───────────────────
        # ───────────── modality‑specific AEs ───────────────────────────────
        # 1) LLM‑derived embedding (512 → 256 → 128)
        self.ae_llm = Enc3Layer(llm_dim, latent_dim=ae_latent, dropout=dropout)

        # mute drug target
        # 2) Drug‑target binary vector: first linear proj → 512, then 512 → 256 → 128
        # self.tgt_proj = nn.Linear(n_genes_target, llm_dim)
        # self.ae_tgt   = Enc3Layer(llm_dim, latent_dim=ae_latent, dropout=dropout)

        # 3) Morgan fingerprint bits (512 → 256 → 128)
        self.ae_fp  = Enc3Layer(fp_bits, latent_dim=ae_latent, dropout=dropout)
        
        
        def mlp_proj(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim)
            )
        
        self.llm_proj_k = mlp_proj(ae_latent, num_heads*d_model)
        self.llm_proj_v = mlp_proj(ae_latent, num_heads*d_model)

        # mute drug target
        # self.tgt_proj_k = mlp_proj(ae_latent, num_heads*d_model)
        # self.tgt_proj_v = mlp_proj(ae_latent, num_heads*d_model)

        self.fp_proj_k  = mlp_proj(ae_latent, num_heads*d_model)
        self.fp_proj_v  = mlp_proj(ae_latent, num_heads*d_model)

        # three cross-attention blocks
        self.xattn1 = nn.MultiheadAttention(d_model, num_heads,
                                            dropout)
        self.xattn2 = nn.MultiheadAttention(d_model, num_heads,
                                            dropout)
        self.xattn3 = nn.MultiheadAttention(d_model, num_heads,
                                            dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

        # ───────────── shared FFN & per-gene head (unchanged) ──────────────
        self.shared_ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model)
        )
        
        # ───────────── NEW: gene‑specific FiLM parameters ──────────────────
        self.gamma_gene = nn.Parameter(torch.ones(max_len, d_model))  # (S,d)
        self.beta_gene  = nn.Parameter(torch.zeros(max_len, d_model)) # (S,d)
        
        # variance-driven FiLM parameters (cell-specific)
        self.var_gamma = nn.Parameter(torch.zeros(max_len, d_model))
        self.var_beta  = nn.Parameter(torch.zeros(max_len, d_model))
        
        self.W = nn.Parameter(torch.empty(max_len, d_model))
        self.b = nn.Parameter(torch.empty(max_len))
        
        self.Wmlp1 = nn.Parameter(torch.empty(max_len, d_model, d_mid))
        self.bmlp1 = nn.Parameter(torch.zeros(max_len, d_mid))
        self.Wmlp2 = nn.Parameter(torch.empty(max_len, d_mid))    # to scalar
        self.bmlp2 = nn.Parameter(torch.zeros(max_len))

        for p in (self.W, self.Wmlp1, self.Wmlp2):
            nn.init.xavier_uniform_(p)
            
        for p in (self.b, self.bmlp1, self.bmlp2):
            nn.init.zeros_(p)
            
        def _make_gate():
            g = nn.Sequential(
                    nn.Linear(2, 16), nn.GELU(),
                    nn.Linear(16, 1),
                    nn.Softplus()        # ≥ 0, smooth
            )
            with torch.no_grad():       # start near zero gain
                g[-2].weight.zero_()
                g[-2].bias.fill_(-4.0)  # Softplus(-4) ≈ 0.018
            return g

        self.gate_attn1 = _make_gate()   # for LLM cross‑attention
        # mute drug target
        # self.gate_attn2 = _make_gate()   # for target cross‑attention
        self.gate_attn3 = _make_gate()   # for FP   cross‑attention
            

    # ----------------------------------------------------------------------
    def _kv(self, proj_k, proj_v, feat):
        """
        proj_* : Linear to (H·D)
        feat   : (B, feat_dim)
        returns k,v with shape (B, H, D)   (seq_len = H tokens)
        """
        B = feat.size(0)
        # (B, H * d_model)  →  reshape →  (H, B, d_model)
        k = proj_k(feat) \
              .view(B, self.num_heads, self.d_model) \
              .permute(1, 0, 2)
        v = proj_v(feat) \
              .view(B, self.num_heads, self.d_model) \
              .permute(1, 0, 2)
        return k, v

    # ----------------------------------------------------------------------
    def forward(self,
                x_base: torch.Tensor,   # (B,S)
                cell_stats: torch.Tensor, # (B,2S)  = [μ ‖ σ²]
                x_llm: torch.Tensor,
                x_fp: torch.Tensor,
                x_tgt: torch.Tensor,
                dose_feat: torch.Tensor, 
                time_feat: torch.Tensor):
        B, S = x_base.shape
        assert S == self.max_len, "gene dim mismatch"

        # prepare 3‑vector per gene ------------------------------------
        mu  = cell_stats[:, :S]
        var = cell_stats[:, S:]
        # concat input features
        x_feat = torch.stack([x_base, mu, var], dim=-1)        # (B,S,3)
    
        # 0) denoise modalities --------------------------------------------
        z_llm = self.ae_llm(x_llm)                       # (B,512)

        # Mute drug target
        # tgt_emb = self.tgt_proj(x_tgt)                           # (B,512)
        # z_tgt = self.ae_tgt(tgt_emb)                    # (B,512)

        z_fp = self.ae_fp(x_fp)                        # (B,512)
        
        # 0.1) prepare the dose and time features
        scalar_pair = torch.stack([dose_feat, time_feat], dim=-1)   # (B,2)
        
        
        # 1) encode genes ----------------------------------------------------
        # --- per‑gene 2‑layer MLP projection ----------------------------------
        h1 = torch.einsum('bsi,sid->bsd', x_feat, self.W1_gene) + self.b1_gene
        h1 = torch.nn.functional.gelu(h1)
        x  = torch.einsum('bsd,sdk->bsk', h1, self.W2_gene) + self.b2_gene
        
        x = x.transpose(0,1)                                    # (S,B,d)
        x = self.gene_encoder(x)

        # 2) genes ⟵ LLM  ----------------------------------------------------
        k1,v1 = self._kv(self.llm_proj_k, self.llm_proj_v, z_llm)
        out,_ = self.xattn1(query=x, key=k1, value=v1)
        
        g1 = self.gate_attn1(scalar_pair).unsqueeze(0)  # (1,B,1)
        x = self.norm1(x + self.dropout(g1 * out))

        # mute drug target
        # 3) genes ⟵ targets  -----------------------------------------------
        # k2,v2     = self._kv(self.tgt_proj_k, self.tgt_proj_v, z_tgt)
        # out,_     = self.xattn2(query=x, key=k2, value=v2)
        
        # g2 = self.gate_attn2(scalar_pair).unsqueeze(0)
        # x = self.norm2(x + self.dropout(g2 * out))

        # 4) genes ⟵ fingerprints  ------------------------------------------
        k3,v3 = self._kv(self.fp_proj_k, self.fp_proj_v, z_fp)
        out,_ = self.xattn3(query=x, key=k3, value=v3)
        
        g3 = self.gate_attn3(scalar_pair).unsqueeze(0)
        x = self.norm3(x + self.dropout(g3 * out))
        
        x = x.transpose(0, 1)                                  # (B,S,d) (back to batch‑first)

        # 4) FFN + per-gene head -------------------------------------------
        h      = self.shared_ffn(x)
        
        # 5) variance-driven FiLM
        var_unsq = var.unsqueeze(-1)  # (B,S,1)
        gamma_v  = 1 + var_unsq * self.var_gamma.unsqueeze(0)
        beta_v   = var_unsq * self.var_beta.unsqueeze(0)
        h = gamma_v * h + beta_v
        
        # 6) Gene‑specific FiLM adaptation ---------------------------------
        h = self.gamma_gene.unsqueeze(0) * h + self.beta_gene.unsqueeze(0)  # (B,S,d)
        
        
        # linear path (unchanged)
        linear_out = torch.einsum('bsd,sd->bs', h, self.W) + self.b      # (B,S) → (B,)

        # residual MLP path
        h_m = torch.einsum('bsd,sdm->bsm', h, self.Wmlp1) + self.bmlp1   # (B,S,d_mid)
        h_m = torch.nn.functional.gelu(h_m)
        mlp_out = torch.einsum('bsm,sm->bs', h_m, self.Wmlp2) + self.bmlp2

        logits = linear_out + mlp_out
        
        return logits


'''
Different Losses
'''
def cosine_loss(output, target):
    cos = nn.CosineSimilarity(dim=1)
    return 1 - cos(output, target).mean()

def correlation_loss(output, target):
    vx = output - output.mean(dim=1, keepdim=True)
    vy = target - target.mean(dim=1, keepdim=True)
    corr = (vx * vy).sum(dim=1) / (torch.norm(vx, dim=1) * torch.norm(vy, dim=1) + 1e-8)
    return 1 - corr.mean()

def mse_plus_cosine(output, target):
    return nn.MSELoss()(output, target) + 0.3 * (cosine_loss(output, target)-1)

def mse_plus_correlation(output, target):
    return nn.MSELoss()(output, target) + 0.3 * (correlation_loss(output, target)-1)


# ---------- sign-flip penalty (flat) ----------------------------
def _sign_penalty(pred: torch.Tensor,
                  target: torch.Tensor,
                  reduction: str = "mean") -> torch.Tensor:
    wrong_sign = (pred * target < 0).float()        # 1 ↔ sign mismatch
    return wrong_sign.mean() if reduction == "mean" else wrong_sign.sum()

# ---------- tanh-alignment loss --------------------------------
def _tanh_alignment(pred: torch.Tensor,
                    target: torch.Tensor,
                    k: float,
                    reduction: str = "mean") -> torch.Tensor:
    diff2 = (torch.tanh(k * pred) - torch.tanh(k * target)).pow(2)
    return diff2.mean() if reduction == "mean" else diff2.sum()

# ---------- factory that produces the combined criterion -----------
def MSE_Sign_loss(λ_mse:   float = 1.0,
                          λ_sign:  float = 0.8,
                          reduction: str = "mean"):
    """
    Returns a function loss(pred, target) compatible with your training loop.
    The default weights assume your plain MSE ≃ 1.  Tweak λ_* to balance terms.
    """
    def _loss(pred, target):
        mse  = F.mse_loss(pred, target, reduction=reduction)
        sign = _sign_penalty(pred, target, reduction=reduction)
        return λ_mse * mse + λ_sign * sign
    return _loss

def MSE_tanh_loss(λ_mse:   float = 1.0,
                          λ_tanh:  float = 1.0,
                          k:       float = 0.5,
                          reduction: str = "mean"):
    """
    Returns a function loss(pred, target) compatible with your training loop.
    The default weights assume your plain MSE ≃ 1.  Tweak λ_* to balance terms.
    """
    def _loss(pred, target):
        mse  = F.mse_loss(pred, target, reduction=reduction)
        tanh = _tanh_alignment(pred, target, k=k, reduction=reduction)
        return λ_mse * mse + λ_tanh * tanh
    return _loss

def MSE_PCC_Sign_loss(λ_mse:  float = 1.0,
                      λ_pcc:  float = 0.3,
                      λ_sign: float = 0.8,
                      reduction: str = "mean"):
    """
    MSE + Pearson-correlation loss + flat sign-flip penalty.
    All weights (λ_*) are tunable so each term lands on a similar scale.
    """
    def _loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse  = F.mse_loss(pred, target, reduction=reduction)
        pcc  = correlation_loss(pred, target)              # ← your helper
        sign = _sign_penalty(pred, target, reduction=reduction)
        return λ_mse * mse + λ_pcc * (pcc-1) + λ_sign * sign
    return _loss


def MSE_PCC_Tanh_loss(λ_mse:   float = 1.0,
                      λ_pcc:   float = 0.3,
                      λ_tanh:  float = 0.6,
                      k:       float = 0.5,
                      reduction: str = "mean"):
    """
    MSE + Pearson-correlation loss + tanh-alignment penalty.
    k controls tanh steepness; λ_* balance the terms.
    """
    def _loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse  = F.mse_loss(pred, target, reduction=reduction)
        pcc  = correlation_loss(pred, target)              # 1 – corr.mean()
        tanh = _tanh_alignment(pred, target, k=k, reduction=reduction)
        return λ_mse * mse + λ_pcc * (pcc-1) + λ_tanh * tanh
    return _loss

loss_options = {
    "MSE": nn.MSELoss(),
    "MAE": nn.L1Loss(),
    "Cosine": cosine_loss,
    "Correlation": correlation_loss,
    "MSE+Cosine": mse_plus_cosine,
    "MSE+Correlation": mse_plus_correlation,
    "MSE+Sign": MSE_Sign_loss(), # uses default λ’s
    "MSE+Tanh": MSE_tanh_loss(), # uses default λ’s
    "MSE+PCC+Sign" : MSE_PCC_Sign_loss(),   # uses default λ’s
    "MSE+PCC+Tanh" : MSE_PCC_Tanh_loss()    # uses default λ’s
}


'''
load in the best model
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GenePerturbationTransformer(d_model=32, num_heads=8, num_encoder_layers=4, dropout=0.1, max_len=978, d_hidden=64, 
                                    n_genes_target=1225, fp_bits=512, llm_dim=512, ae_latent=128)
model = model.to(device)



model_dir = Path("./Model")
## change the ckpt_name correspondingly if you are training your own model.
ckpt_name = f"transformer_d32h8l4_{args.split_type}_dp1_MSECor_lr1_CosSche_3XAttn_sepGene2EncPred_newAttn_FiLM_Enc3DimReducLa128_CellAware_frontEndMLPsimp_whole_DoseTimeAsScalarXattns_LLMMFP_first50epoch.pth"
ckpt_path = model_dir / ckpt_name

# optional: check it exists
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

checkpoint = torch.load(ckpt_path, map_location=device)

print(checkpoint["epoch"])
model.load_state_dict(checkpoint['model_state'] 
                       if 'model_state' in checkpoint else checkpoint)


def eval_model(model, dataloader, criterion, device):
    model.eval()
    running_loss, all_true, all_pred, all_true_delta, all_pred_delta = 0.0, [], [], [], []

    with torch.no_grad():
        for x_base, cell_stats, x_llm, x_fp, x_tgt, y_true, dose, time in dataloader:  # ← correct order
            x_base = x_base.to(device)
            cell_stats = cell_stats.to(device)
            x_llm  = x_llm.to(device)
            x_tgt  = x_tgt.to(device)
            x_fp   = x_fp.to(device)
            dose   = dose.to(device)
            time   = time.to(device)
            y_true = y_true.to(device)

            y_pred = model(x_base, cell_stats, x_llm, x_fp, x_tgt, dose, time)
            
            delta_pred = y_pred - x_base
            delta_true = y_true - x_base
            
            loss = criterion(delta_pred, delta_true)

            running_loss += loss.item() * x_base.size(0)
            all_true.append(y_true.cpu())
            all_pred.append(y_pred.cpu())
            
            all_true_delta.append(delta_true.cpu())
            all_pred_delta.append(delta_pred.cpu())

    epoch_loss = running_loss / len(dataloader.dataset)

    all_y_true = torch.cat(all_true)  # Shape: (n_samples, 978)
    all_y_pred = torch.cat(all_pred)  # Shape: (n_samples, 978)
    
    all_delta_true = torch.cat(all_true_delta)
    all_delta_pred = torch.cat(all_pred_delta)
    
    
    # Average R² across samples
    r2_list = []
    pcc_list = []
    r2_delta_list = []
    pcc_delta_list = []
    for i in range(all_y_true.shape[0]):
        true_sample = all_y_true[i].numpy()
        pred_sample = all_y_pred[i].numpy()
        
        true_sample_delta = all_delta_true[i].numpy()
        pred_sample_delta = all_delta_pred[i].numpy()

        # R²
        r2_val = r2_score(true_sample, pred_sample)
        r2_list.append(r2_val)
        r2_val_delta = r2_score(true_sample_delta, pred_sample_delta)
        r2_delta_list.append(r2_val_delta)

        # PCC
        if np.std(true_sample) > 1e-6 and np.std(pred_sample) > 1e-6:
            pcc_val, _ = pearsonr(true_sample, pred_sample)
            pcc_list.append(pcc_val)
        else:
            pcc_list.append(0.0)  # or np.nan, depending on how you want to handle flat vectors
            
        # PCC
        if np.std(true_sample_delta) > 1e-6 and np.std(pred_sample_delta) > 1e-6:
            pcc_val_delta, _ = pearsonr(true_sample_delta, pred_sample_delta)
            pcc_delta_list.append(pcc_val_delta)
        else:
            pcc_delta_list.append(0.0)  # or np.nan, depending on how you want to handle flat vectors
    
    mse_loss = torch.mean((all_y_true - all_y_pred) ** 2).item()
    
    avg_r2 = np.mean(r2_list)
    avg_pcc = np.mean(pcc_list)

    avg_delta_r2 = np.mean(r2_delta_list)
    avg_delta_pcc = np.mean(pcc_delta_list)

    return epoch_loss, avg_r2, mse_loss, avg_pcc, avg_delta_r2, avg_delta_pcc

def pred_model(model, dataloader, device):
    """
    Evaluate gene-specific metrics on the validation set.
    
    For each gene (across all samples):
      - Computes the squared errors and calculates their mean (MSE Mean)
        and standard deviation (MSE Std).
      - Computes a single R² score over all samples.
      - Computes the Pearson correlation coefficient over all samples.
    
    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): Validation DataLoader.
        device (torch.device): The device (CPU/GPU).
    
    Returns:
        mse_df (pd.DataFrame): DataFrame with columns ['Gene', 'MSE Mean', 'MSE Std'],
                               sorted in ascending order by 'MSE Mean'.
        r2_df (pd.DataFrame): DataFrame with columns ['Gene', 'R2'],
                              sorted in descending order by 'R2'.
        pearson_df (pd.DataFrame): DataFrame with columns ['Gene', 'Pearson'],
                                   sorted in descending order by 'Pearson'.
    """
    # Set the model to evaluation mode.
    model.eval()
    all_baseline = []
    all_y_true = []
    all_y_pred = []
    all_true_delta, all_pred_delta = [], []
    

    # Collect predictions and ground truth values.
    with torch.no_grad():
        for x_base, cell_stats, x_llm, x_fp, x_tgt, y_true, dose, time in dataloader:  # ← correct order
            x_base = x_base.to(device)
            cell_stats = cell_stats.to(device)
            x_llm  = x_llm.to(device)
            x_tgt  = x_tgt.to(device)
            x_fp   = x_fp.to(device)
            dose   = dose.to(device)
            time   = time.to(device)
            y_true = y_true.to(device)

            y_pred = model(x_base, cell_stats, x_llm, x_fp, x_tgt, dose, time)
            
            delta_pred = y_pred - x_base
            delta_true = y_true - x_base
            
            all_baseline.append(x_base.cpu())
            all_y_true.append(y_true.cpu())
            all_y_pred.append(y_pred.cpu())
            
            all_true_delta.append(delta_true.cpu())
            all_pred_delta.append(delta_pred.cpu())


    # --- concatenate the batch lists ---
    baseline_mat = torch.cat(all_baseline).numpy()   # shape: (n_samples, n_genes)
    true_mat     = torch.cat(all_y_true).numpy()     # same shape
    pred_mat     = torch.cat(all_y_pred).numpy()     # same shape
    all_delta_true_mat = torch.cat(all_true_delta).numpy()
    all_delta_pred_mat = torch.cat(all_pred_delta).numpy()

    # optional: keep track of which AnnData row each sample came from
    row_order = dataloader.dataset.indices           # numpy array of adata row positions

    # --- build the DataFrame ---
    df_preds = pd.DataFrame({
        'adata_row' : row_order,          # or adata.obs_names[row_order] for IDs
        'Baseline'  : list(baseline_mat), # store each vector as a single cell (dtype=object)
        'True'      : list(true_mat),
        'Pred'      : list(pred_mat),
        'True_delta': list(all_delta_true_mat),
        'Pred_delta': list(all_delta_pred_mat)
    })

    return df_preds


loss_name = "MSE+Correlation"
loss_fn = loss_options[loss_name]

results_dir = f'./Results/PredictionResults/pred_df_{args.split_type}'
out_dir = Path(results_dir)
out_dir.mkdir(parents=True, exist_ok=True)   # create folders if missing
out_csv = out_dir / f"pred_df_{args.subset}.csv"

if act_stat:
    if validation_pred:
        epoch_loss, avg_r2, mse_loss, avg_pcc, avg_delta_r2, avg_delta_pcc = eval_model(model, valid_loader, loss_fn, device)
    else: 
        epoch_loss, avg_r2, mse_loss, avg_pcc, avg_delta_r2, avg_delta_pcc = eval_model(model, test_loader, loss_fn, device)

if act_pred:
    if validation_pred:
        pred_df = pred_model(model, valid_loader, device)
        pred_df.to_csv(out_csv, index=False, header=True)
        print(f"Saved to {out_csv}")
    else: 
        pred_df = pred_model(model, test_loader, device)
        pred_df.to_csv(out_csv, index=False, header=True)
        print(f"Saved to {out_csv}")

if act_both:
    if validation_pred:
        epoch_loss, avg_r2, mse_loss, avg_pcc, avg_delta_r2, avg_delta_pcc = eval_model(model, valid_loader, loss_fn, device)
        pred_df = pred_model(model, valid_loader, device)
        pred_df.to_csv(out_csv, index=False, header=True)
        print(f"Saved to {out_csv}")
    else: 
        epoch_loss, avg_r2, mse_loss, avg_pcc, avg_delta_r2, avg_delta_pcc = eval_model(model, test_loader, loss_fn, device)
        pred_df = pred_model(model, test_loader, device)
        pred_df.to_csv(out_csv, index=False, header=True)
        print(f"Saved to {out_csv}")


print(f"Test Loss: {epoch_loss:.4f}, Test MSE: {mse_loss:.4f}, Test R2: {avg_r2:.4f}, Test PCC: {avg_pcc:.4f}, Test delta R2: {avg_delta_r2:.4f}, Test delta PCC: {avg_delta_pcc:.4f}")

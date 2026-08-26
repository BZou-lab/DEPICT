#!/usr/bin/env python3
"""
DEPICT Analysis 3 — non-interactive compute stage.

Run under Slurm with 8 CPUs, e.g.
  python DEPICT_Analysis3_Compute.py --n-workers 8

This script produces all durable tables needed by the companion plotting notebook:
  audit/, scores/, metrics/, null/, mechanism/, examples/.

Important GSEA implementation note:
- Global scoring uses deterministic *signed weighted preranked GSEA* for every
  observed/predicted DGE profile, exactly as locked in Addendum v5.
- Genes are ranked from largest positive to largest negative signed DGE.
- Hit weights are abs(DGE)^p with p=1.0; misses have equal decrement.
- Stable mergesort is the locked tie rule.
- Example-only conventional permutation preranked GSEA uses gene-set
  permutations, computes nominal p-values, normalized ES (NES), and BH FDR
  within each example/source combination.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse, hashlib, json, math, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

PROJECT_ROOT = Path("~/DEPICT")
MAIN_WORK_DIR = PROJECT_ROOT / "Code/downstream_analysis_code/PathwayRecovery"

CONFIG = {
    "adata_path": PROJECT_ROOT / "Data/FinalData/adataAfterClean.h5ad",
    "predicted_dge_dir": MAIN_WORK_DIR / "predicted_dge",
    "output_dir": MAIN_WORK_DIR / "analysis3_pathway_recovery_revised",
    "hallmark_gmt": MAIN_WORK_DIR / "data/h.all.v2026.1.Hs.symbols.gmt.txt",
    "mechanism_panel_gmt": MAIN_WORK_DIR / "data/depict_analysis3_mechanism_panel.v1.Hs.symbols.gmt",
    "drug_class_csv": MAIN_WORK_DIR / "data/depict_analysis3_drug_classes.v2.csv",
    "drug_splits": [f"drug_split{i}" for i in range(1, 6)],
    "cell_splits": [f"cell_split{i}" for i in range(1, 6)],
    "min_landmark_overlap": 10,
    "top_ks": (3, 5),
    "strong_es_threshold": 0.20,
    "score_chunk_rows": 512,
    "gsea_weight_exponent": 1.0,
    "tie_rule": "stable_mergesort_descending_signed_dge",
    "dose_bin_widths_um": (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
    "min_null_alternative_drugs": 2,
    "null_draws_full": 500,
    "null_calibration_draws": (200, 500, 1000),
    "null_calibration_max_profiles_per_split": 200,
    "example_classes": ("PI3K_AKT_MTOR", "CDK", "TOPOISOMERASE", "RTK_SIGNALING"),
    "eligible_mechanism_classes": ("PI3K_AKT_MTOR", "CDK", "TOPOISOMERASE", "HDAC", "RTK_SIGNALING"),
    "min_profiles_for_example_drug": 3,
    "example_n_permutations": 1000,
    "random_seed": 6666,
}
OUT = Path(CONFIG["output_dir"])
for d in ["audit","scores","metrics","null","mechanism","examples","manifests"]:
    (OUT/d).mkdir(parents=True, exist_ok=True)

def require(x, msg):
    if not x: raise RuntimeError(msg)

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:4],"big")

def hash_sequence(values: Iterable[object]) -> str:
    h=hashlib.sha256()
    for v in values:
        b=str(v).encode(); h.update(len(b).to_bytes(8,"big")); h.update(b)
    return h.hexdigest()

def decode(values):
    return np.asarray([x.decode() if isinstance(x,(bytes,np.bytes_)) else str(x) for x in values],dtype=object)

def read_gmt(path: Path):
    require(path.exists(),f"Missing GMT: {path}")
    out={}
    for i,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        f=line.split("\t"); require(len(f)>=3,f"Malformed GMT line {i}")
        out[f[0].strip()]=list(dict.fromkeys(g.strip().upper() for g in f[2:] if g.strip()))
    return out

def make_index(gsets, var_names):
    lookup={str(g).upper():i for i,g in enumerate(var_names)}
    idx={}; audit=[]
    for name, genes in gsets.items():
        ii=np.array(sorted({lookup[g] for g in genes if g in lookup}),dtype=np.int32)
        keep=len(ii)>=CONFIG["min_landmark_overlap"]
        audit.append({"pathway":name,"n_input_genes":len(genes),"n_landmark_overlap":len(ii),"retained":keep,
                      "matched_landmark_genes":";".join(map(str,var_names[ii]))})
        if keep: idx[name]=ii
    require(len(idx)>=max(CONFIG["top_ks"]),"Too few eligible pathways.")
    return idx,pd.DataFrame(audit).sort_values("pathway")

# Globals initialized once in parent then inherited by Linux fork workers.
adata=ad.read_h5ad(CONFIG["adata_path"],backed="r")
VAR_NAMES=pd.Index(adata.var_names.astype(str)); ADATA_HASH=hash_sequence(VAR_NAMES)
require(len(VAR_NAMES)==978,f"Expected 978 genes, got {len(VAR_NAMES)}")
HALLMARK_INDEX,HALLMARK_AUDIT=make_index(read_gmt(Path(CONFIG["hallmark_gmt"])),VAR_NAMES)
PANEL_INDEX,PANEL_AUDIT=make_index(read_gmt(Path(CONFIG["mechanism_panel_gmt"])),VAR_NAMES)
HALLMARK_NAMES=list(HALLMARK_INDEX); PANEL_NAMES=list(PANEL_INDEX)

def weighted_signed_es_matrix(dge: np.ndarray, pathway_index: Mapping[str,np.ndarray]) -> np.ndarray:
    """Deterministic signed weighted preranked GSEA ES, with p=CONFIG['gsea_weight_exponent']."""
    x=np.asarray(dge,dtype=np.float32); n_rows,n_genes=x.shape
    require(n_genes==len(VAR_NAMES),"Gene dimension mismatch.")
    order=np.argsort(-x,axis=1,kind="mergesort")
    rank_abs=np.take_along_axis(np.abs(x),order,axis=1) ** float(CONFIG["gsea_weight_exponent"])
    out=np.empty((n_rows,len(pathway_index)),dtype=np.float32)
    for j,idx in enumerate(pathway_index.values()):
        hit_mask=np.zeros(n_genes,dtype=np.float32); hit_mask[idx]=1
        hits=hit_mask[order]
        hit_weights=hits*rank_abs
        denom=hit_weights.sum(axis=1,keepdims=True)
        # mathematically valid only if >=1 pathway member; every retained set has >=10
        hit_step=np.divide(hit_weights,denom,out=np.zeros_like(hit_weights),where=denom>0)
        miss_step=(1.0-hits)/float(n_genes-len(idx))
        running=np.cumsum(hit_step-miss_step,axis=1)
        mx=running.max(axis=1); mn=running.min(axis=1)
        out[:,j]=np.where(np.abs(mx)>=np.abs(mn),mx,mn)
    return out

def h5_path(split): return Path(CONFIG["predicted_dge_dir"])/split/"predicted_dge_test.h5"

def validate_h5(h5,path):
    need={"obs_name","pert_iname","cell_id","dose","pert_time","observed_dge","predicted_dge"}
    require(need.issubset(h5.keys()),f"{path} missing {need-set(h5.keys())}")
    require(h5["observed_dge"].shape==h5["predicted_dge"].shape,"obs/pred shape mismatch")
    saved=h5.attrs.get("adata_var_names_hash_sha256")
    if isinstance(saved,bytes): saved=saved.decode()
    require(str(saved)==ADATA_HASH,f"Gene hash mismatch: {path}")

def score_split(split, library, index):
    out=OUT/"scores"/f"{library}__{split}__profile_scores.parquet"
    # Reuse only a scored file that records the weighted-GSEA implementation.
    manifest=OUT/"scores"/f"{library}__{split}__scoring_manifest.json"
    if out.exists() and manifest.exists():
        meta=json.loads(manifest.read_text())
        if meta.get("gsea_weight_exponent")==CONFIG["gsea_weight_exponent"] and meta.get("tie_rule")==CONFIG["tie_rule"]:
            return out
    chunks=[]; p=h5_path(split); t=time.perf_counter()
    with h5py.File(p,"r") as h:
        validate_h5(h,p); n=h["observed_dge"].shape[0]; names=list(index)
        for s in range(0,n,CONFIG["score_chunk_rows"]):
            e=min(s+CONFIG["score_chunk_rows"],n)
            obs=weighted_signed_es_matrix(h["observed_dge"][s:e],index)
            pred=weighted_signed_es_matrix(h["predicted_dge"][s:e],index)
            base=pd.DataFrame({"split_type":split,"obs_name":decode(h["obs_name"][s:e]),"pert_iname":decode(h["pert_iname"][s:e]),
                               "cell_id":decode(h["cell_id"][s:e]),"dose":np.asarray(h["dose"][s:e],np.float32),
                               "pert_time":np.asarray(h["pert_time"][s:e],np.float32)})
            chunks.append(pd.concat([base,pd.DataFrame(obs,columns=[f"obs_es__{z}" for z in names]),
                                     pd.DataFrame(pred,columns=[f"pred_es__{z}" for z in names])],axis=1))
    ans=pd.concat(chunks,ignore_index=True); require(ans.obs_name.is_unique,f"Duplicate obs_name in {split}")
    ans.to_parquet(out,index=False)
    manifest.write_text(json.dumps({"split_type":split,"library":library,"gsea_weight_exponent":CONFIG["gsea_weight_exponent"],
        "tie_rule":CONFIG["tie_rule"],"n_profiles":len(ans),"adata_var_names_hash_sha256":ADATA_HASH},indent=2))
    print(f"[{split}|{library}] scored {len(ans):,} profiles in {(time.perf_counter()-t)/60:.1f} min",flush=True)
    return out

def corr_rows_against_vector(candidate: np.ndarray, focal: np.ndarray) -> np.ndarray:
    """Vectorized Spearman correlations of candidate rows vs focal pathway vector."""
    fr=rankdata(focal,method="average"); fr=fr-fr.mean(); fs=np.sqrt(np.sum(fr*fr))
    cr=np.apply_along_axis(rankdata,1,candidate,method="average"); cr=cr-cr.mean(axis=1,keepdims=True)
    denom=np.sqrt(np.sum(cr*cr,axis=1))*fs
    return np.divide(cr@fr,denom,out=np.full(len(candidate),np.nan),where=denom>0)

def build_candidate_cache(scores, observed_cols):
    """Cache collapsed alternative-drug observed ES pools by (cell,time,focal_dose)."""
    cache={}
    for (cell,t),g in scores.groupby(["cell_id","pert_time"],sort=False):
        gg=g[["pert_iname","dose",*observed_cols]]
        for dose in np.unique(gg["dose"].to_numpy(float)):
            key=(cell,float(t),float(dose)); chosen=None; chosen_w=np.nan
            for width in CONFIG["dose_bin_widths_um"]:
                c=gg[np.abs(gg["dose"].to_numpy(float)-dose)<=width/2].groupby("pert_iname",as_index=False)[observed_cols].mean()
                # retain full pool; focal drug exclusion happens later
                if len(c)>=CONFIG["min_null_alternative_drugs"]+1:
                    chosen=c; chosen_w=float(width); break
            cache[key]=(chosen,chosen_w)
    return cache

def null_one(scores, cache, i, observed_cols, predicted_cols, draws, seed):
    r=scores.iloc[i]; pool,width=cache.get((r.cell_id,float(r.pert_time),float(r.dose)),(None,np.nan))
    focal_obs=r[observed_cols].to_numpy(float); focal_pred=r[predicted_cols].to_numpy(float)
    r_dep=float(corr_rows_against_vector(focal_pred[None,:],focal_obs)[0])
    if pool is None: return (0,width,0,r_dep,np.nan,np.nan)
    c=pool.loc[pool.pert_iname!=r.pert_iname,observed_cols].to_numpy(float)
    m=len(c)
    if m<CONFIG["min_null_alternative_drugs"] or not np.isfinite(r_dep): return (m,width,0,r_dep,np.nan,np.nan)
    r_all=corr_rows_against_vector(c,focal_obs)
    rng=np.random.default_rng(seed)
    # MC draws with replacement, as locked; vectorized after candidate-pool caching.
    null_mean=float(np.nanmean(r_all[rng.integers(0,m,size=draws)]))
    return (m,width,draws,r_dep,null_mean,float(r_dep-null_mean))

def run_null(scores, names, split):
    obs=[f"obs_es__{x}" for x in names]; pred=[f"pred_es__{x}" for x in names]
    t=time.perf_counter(); cache=build_candidate_cache(scores,obs); rows=[]
    for i in range(len(scores)):
        m,w,n,rd,rn,g=null_one(scores,cache,i,obs,pred,CONFIG["null_draws_full"],
                               stable_seed(CONFIG["random_seed"],split,scores.iloc[i].obs_name,"full_null"))
        r=scores.iloc[i]
        rows.append((split,r.obs_name,r.pert_iname,r.cell_id,r.dose,r.pert_time,m,w,n,rd,rn,g))
    out=pd.DataFrame(rows,columns=["split_type","obs_name","pert_iname","cell_id","dose","pert_time","candidate_pool_size",
        "chosen_dose_bin_width_um","n_sampled_alternatives","r_depict","rbar_null_mc","gain"])
    print(f"[{split}] null B={CONFIG['null_draws_full']} done in {(time.perf_counter()-t)/60:.1f} min",flush=True)
    return out,cache

def run_calibration(scores,names,split,cache):
    obs=[f"obs_es__{x}" for x in names]; pred=[f"pred_es__{x}" for x in names]
    eligible=[i for i in range(len(scores)) if (lambda z: z is not None and len(z.loc[z.pert_iname!=scores.iloc[i].pert_iname])>=CONFIG["min_null_alternative_drugs"])(cache.get((scores.iloc[i].cell_id,float(scores.iloc[i].pert_time),float(scores.iloc[i].dose)),(None,np.nan))[0])]
    eligible=sorted(eligible,key=lambda i:str(scores.iloc[i].obs_name))[:CONFIG["null_calibration_max_profiles_per_split"]]
    rows=[]
    for i in eligible:
        r=scores.iloc[i]
        for B in CONFIG["null_calibration_draws"]:
            m,w,n,rd,rn,g=null_one(scores,cache,i,obs,pred,B,stable_seed(CONFIG["random_seed"],split,r.obs_name,B))
            rows.append((split,r.obs_name,B,stable_seed(CONFIG["random_seed"],split,r.obs_name,B),m,w,n,rd,rn,g))
    return pd.DataFrame(rows,columns=["split_type","obs_name","draws","seed","candidate_pool_size","chosen_dose_bin_width_um",
                                      "n_sampled_alternatives","r_depict","rbar_null_mc","gain"])

def profile_metrics(scores,names):
    O=scores[[f"obs_es__{x}" for x in names]].to_numpy(float); P=scores[[f"pred_es__{x}" for x in names]].to_numpy(float)
    rows=[]
    for i in range(len(scores)):
        strong=np.abs(O[i])>=CONFIG["strong_es_threshold"]
        pr=corr_rows_against_vector(P[i:i+1],O[i])[0]
        d={"pathway_profile_concordance":pr,"n_observed_strong_pathways":int(strong.sum()),
           "directional_agreement":float(np.mean(np.sign(P[i,strong])==np.sign(O[i,strong]))) if strong.any() else np.nan}
        for k in CONFIG["top_ks"]:
            for sign,label in [(1,"positive"),(-1,"negative")]:
                a=np.argsort(-(sign*P[i]),kind="mergesort")[:k]; b=np.argsort(-(sign*O[i]),kind="mergesort")[:k]
                d[f"{label}_recall_at_{k}"]=len(set(a)&set(b))/k
        rows.append(d)
    return pd.concat([scores[["split_type","obs_name","pert_iname","cell_id","dose","pert_time"]].reset_index(drop=True),pd.DataFrame(rows)],axis=1)

def aggregate_units(pm,unit):
    metric=[c for c in pm.columns if c not in ["split_type","obs_name","pert_iname","cell_id","dose","pert_time"]]
    return pm.groupby(["split_type",unit],as_index=False)[metric].mean()

def fold_summary(units,cols):
    return units.groupby("split_type",as_index=False)[cols].agg(["mean","median","count"]).reset_index()

def worker(split, kind):
    index=HALLMARK_INDEX if kind=="hallmark" else PANEL_INDEX; names=list(index)
    p=score_split(split,kind,index); scores=pd.read_parquet(p)
    if kind=="mechanism": return {"split":split,"score":str(p)}
    pm=profile_metrics(scores,names); pm.to_parquet(OUT/"metrics"/f"hallmark__{split}__profile_metrics.parquet",index=False)
    unit="pert_iname" if split.startswith("drug_") else "cell_id"
    um=aggregate_units(pm,unit); um.to_csv(OUT/"metrics"/f"hallmark__{split}__{unit}_metrics.csv",index=False)
    null,cache=run_null(scores,names,split); null.to_parquet(OUT/"null"/f"hallmark__{split}__profile_null.parquet",index=False)
    cal=run_calibration(scores,names,split,cache); cal.to_csv(OUT/"null"/f"hallmark__{split}__null_calibration.csv",index=False)
    return {"split":split,"profile":str(OUT/"metrics"/f"hallmark__{split}__profile_metrics.parquet"),
            "unit":str(OUT/"metrics"/f"hallmark__{split}__{unit}_metrics.csv"),"null":str(OUT/"null"/f"hallmark__{split}__profile_null.parquet"),
            "calibration":str(OUT/"null"/f"hallmark__{split}__null_calibration.csv")}

def load_classes():
    x=pd.read_csv(CONFIG["drug_class_csv"]); x["pert_iname"]=x.pert_iname.astype(str)
    return x.loc[x.eligible_for_class_level_analysis.astype(str).str.lower().isin(["true","1","yes"]) &
                 x.drug_class.isin(CONFIG["eligible_mechanism_classes"]),["pert_iname","drug_class"]].drop_duplicates()

def choose_examples(panel_scores, classes):
    c=(panel_scores.merge(classes,on="pert_iname").groupby(["split_type","drug_class","pert_iname"],as_index=False)
       .agg(n_profiles=("obs_name","size")))
    rows=[]
    for cls in CONFIG["example_classes"]:
        z=c[(c.drug_class==cls)&(c.n_profiles>=CONFIG["min_profiles_for_example_drug"])].copy()
        if z.empty: rows.append({"drug_class":cls,"selected":False}); continue
        z["rank"]=z.split_type.map({s:i for i,s in enumerate(CONFIG["drug_splits"])}).fillna(999)
        q=z.sort_values(["rank","pert_iname"],kind="mergesort").iloc[0]
        rows.append({"drug_class":cls,"selected":True,"split_type":q.split_type,"pert_iname":q.pert_iname,"n_profiles_for_drug_fold":q.n_profiles})
    return pd.DataFrame(rows)

def fetch_dge(split, names):
    wanted=set(names); rows=[]
    with h5py.File(h5_path(split),"r") as h:
        decoded=decode(h["obs_name"][:]); pos=np.where(np.isin(decoded,list(wanted)))[0]
        for i in pos:
            rows.append({"obs_name":decoded[i],"observed_dge":np.asarray(h["observed_dge"][i],np.float32),"predicted_dge":np.asarray(h["predicted_dge"][i],np.float32)})
    return pd.DataFrame(rows)

def gsea_one(x,index,B,seed):
    order=np.argsort(-x,kind="mergesort"); rank_abs=np.abs(x[order])**CONFIG["gsea_weight_exponent"]; n=len(x); rng=np.random.default_rng(seed); rows=[]
    for p,idx in index.items():
        def es(pos):
            h=np.zeros(n,dtype=float); h[pos]=1; hw=h*rank_abs; run=np.cumsum(hw/hw.sum()-(1-h)/(n-len(pos))); mx,mn=run.max(),run.min(); return mx if abs(mx)>=abs(mn) else mn
        obs=es(idx); null=np.array([es(rng.choice(n,len(idx),replace=False)) for _ in range(B)])
        same=null[null>=0] if obs>=0 else -null[null<0]
        nes=obs/(np.mean(np.abs(same)) if len(same) else np.nan)
        pval=(1+np.sum(np.abs(null)>=abs(obs)))/(B+1)
        rows.append((p,obs,nes,pval))
    ans=pd.DataFrame(rows,columns=["pathway","es","nes","nominal_p_value"])
    ans["fdr_bh"]=multipletests(ans.nominal_p_value,method="fdr_bh")[1]
    return ans

def postprocess(hallmark_done, panel_done):
    # consolidated Hallmark artifacts
    for label,splits,unit in [("drug_split_primary",CONFIG["drug_splits"],"pert_iname"),("cell_split_replication",CONFIG["cell_splits"],"cell_id")]:
        pm=pd.concat([pd.read_parquet(OUT/"metrics"/f"hallmark__{s}__profile_metrics.parquet") for s in splits],ignore_index=True)
        um=pd.concat([pd.read_csv(OUT/"metrics"/f"hallmark__{s}__{unit}_metrics.csv") for s in splits],ignore_index=True)
        nl=pd.concat([pd.read_parquet(OUT/"null"/f"hallmark__{s}__profile_null.parquet") for s in splits],ignore_index=True)
        ca=pd.concat([pd.read_csv(OUT/"null"/f"hallmark__{s}__null_calibration.csv") for s in splits],ignore_index=True)
        pm.to_parquet(OUT/"metrics"/f"hallmark__{label}__all_profile_metrics.parquet",index=False); um.to_csv(OUT/"metrics"/f"hallmark__{label}__all_unit_metrics.csv",index=False)
        nl.to_parquet(OUT/"null"/f"hallmark__{label}__all_profile_null.parquet",index=False); ca.to_csv(OUT/"null"/f"hallmark__{label}__null_calibration_all.csv",index=False)
        gain=nl.groupby(["split_type",unit],as_index=False).agg(gain=("gain","mean"),r_depict=("r_depict","mean"),rbar_null_mc=("rbar_null_mc","mean"),n_profiles=("obs_name","size"))
        gain.to_csv(OUT/"null"/f"hallmark__{label}__unit_gain.csv",index=False)
        cov=nl.assign(eligible=nl.candidate_pool_size>=CONFIG["min_null_alternative_drugs"]).groupby("split_type",as_index=False).agg(n_profiles=("obs_name","size"),n_eligible_profiles=("eligible","sum"),eligible_profile_fraction=("eligible","mean"),median_candidate_pool=("candidate_pool_size","median"))
        cov.to_csv(OUT/"null"/f"hallmark__{label}__null_coverage_by_fold.csv",index=False)
    # mechanism summaries / locked examples / example GSEA
    classes=load_classes(); scores=pd.concat([pd.read_parquet(OUT/"scores"/f"mechanism__{s}__profile_scores.parquet") for s in CONFIG["drug_splits"]+CONFIG["cell_splits"]],ignore_index=True)
    scores.to_parquet(OUT/"mechanism"/"mechanism_panel_all_profile_scores.parquet",index=False)
    obs_cols=[f"obs_es__{x}" for x in PANEL_NAMES]; pred_cols=[f"pred_es__{x}" for x in PANEL_NAMES]
    long=[]
    for source,cols in [("observed",obs_cols),("predicted",pred_cols)]:
        z=scores[["split_type","pert_iname"]+cols].merge(classes,on="pert_iname")
        z=z.groupby(["split_type","pert_iname","drug_class"],as_index=False)[cols].mean()
        z=z.melt(id_vars=["split_type","pert_iname","drug_class"],var_name="pathway",value_name="es"); z["source"]=source; z["pathway"]=z.pathway.str.replace("obs_es__","",regex=False).str.replace("pred_es__","",regex=False)
        long.append(z)
    drug_long=pd.concat(long,ignore_index=True); drug_long.to_csv(OUT/"mechanism"/"mechanism_panel_drug_level_scores.csv",index=False)
    cls=drug_long.groupby(["split_type","drug_class","pathway","source"],as_index=False).agg(n_drugs=("pert_iname","nunique"),mean_es=("es","mean"),sd_es=("es","std"))
    cls.to_csv(OUT/"mechanism"/"mechanism_panel_class_level_summary.csv",index=False)
    # Drug-level bootstrap CIs, preserving fold/class/pathway/source strata.
    boot_rows=[]; rng=np.random.default_rng(CONFIG["random_seed"])
    for keys,g in drug_long.groupby(["split_type","drug_class","pathway","source"],sort=True):
        vals=g["es"].dropna().to_numpy(float)
        if len(vals)==0: continue
        means=np.mean(rng.choice(vals,size=(2000,len(vals)),replace=True),axis=1)
        boot_rows.append({"split_type":keys[0],"drug_class":keys[1],"pathway":keys[2],"source":keys[3],
                          "n_drugs":len(vals),"mean_es":float(vals.mean()),
                          "ci95_low":float(np.quantile(means,.025)),"ci95_high":float(np.quantile(means,.975)),
                          "n_bootstrap":2000})
    pd.DataFrame(boot_rows).to_csv(OUT/"mechanism"/"mechanism_panel_class_level_bootstrap_ci.csv",index=False)
    # Omnibus class-label permutation within each fold/pathway/source, only when >=2 classes have >=3 drugs.
    perm_rows=[]; rng=np.random.default_rng(CONFIG["random_seed"])
    for keys,g in drug_long.groupby(["split_type","pathway","source"],sort=True):
        counts=g["drug_class"].value_counts(); keep=counts[counts>=3].index
        z=g[g["drug_class"].isin(keep)][["drug_class","es"]].dropna()
        if z["drug_class"].nunique()<2: continue
        obs=float(np.var(z.groupby("drug_class")["es"].mean().to_numpy(),ddof=0))
        labels=z["drug_class"].to_numpy(); values=z["es"].to_numpy(float); stats=np.empty(5000)
        for b in range(5000):
            stats[b]=np.var(pd.DataFrame({"c":rng.permutation(labels),"v":values}).groupby("c")["v"].mean().to_numpy(),ddof=0)
        perm_rows.append({"split_type":keys[0],"pathway":keys[1],"source":keys[2],"n_drugs":len(z),
                          "n_classes":z["drug_class"].nunique(),"between_class_variance":obs,
                          "permutation_p_value":float((1+np.sum(stats>=obs))/5001),"n_permutations":5000})
    pd.DataFrame(perm_rows).to_csv(OUT/"mechanism"/"mechanism_panel_class_label_permutation.csv",index=False)
    ex=choose_examples(scores[scores.split_type.isin(CONFIG["drug_splits"])],classes); ex.to_csv(OUT/"examples"/"figure4d_locked_example_selection.csv",index=False)
    selected=[]
    for _,r in ex[ex.selected==True].iterrows():
        z=scores[(scores.split_type==r.split_type)&(scores.pert_iname==r.pert_iname)].sort_values(["cell_id","pert_time","dose","obs_name"],kind="mergesort").iloc[0]
        q=r.to_dict(); q.update({"representative_obs_name":z.obs_name,"cell_id":z.cell_id,"dose":z.dose,"pert_time":z.pert_time}); selected.append(q)
    ex2=pd.DataFrame(selected); ex2.to_csv(OUT/"examples"/"figure4d_selected_examples_with_context.csv",index=False)
    rows=[]
    for split,g in ex2.groupby("split_type"):
        d=fetch_dge(split,g.representative_obs_name.tolist())
        for _,r in g.iterrows():
            rr=d[d.obs_name==r.representative_obs_name].iloc[0]
            for source,vec in [("observed",rr.observed_dge),("predicted",rr.predicted_dge)]:
                tab=gsea_one(vec,PANEL_INDEX,CONFIG["example_n_permutations"],stable_seed(CONFIG["random_seed"],r.representative_obs_name,source))
                for k,v in r.items(): tab[k]=v
                tab["source"]=source; rows.append(tab)
    pd.concat(rows,ignore_index=True).to_csv(OUT/"examples"/"figure4d_example_permutation_gsea_within_example_fdr.csv",index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n-workers",type=int,default=8); args=ap.parse_args()
    CONFIG["n_workers"]=max(1,args.n_workers)
    HALLMARK_AUDIT.assign(library="hallmark").to_csv(OUT/"audit"/"hallmark_gene_mapping_audit.csv",index=False)
    PANEL_AUDIT.assign(library="mechanism_panel").to_csv(OUT/"audit"/"mechanism_panel_gene_mapping_audit.csv",index=False)
    pd.DataFrame([{"adata_var_names_hash_sha256":ADATA_HASH,"gsea_weight_exponent":CONFIG["gsea_weight_exponent"],"tie_rule":CONFIG["tie_rule"],
                   "null_draws_full":CONFIG["null_draws_full"],"n_workers":CONFIG["n_workers"]}]).to_csv(OUT/"audit"/"analysis3_locked_configuration.csv",index=False)
    jobs=[(s,"hallmark") for s in CONFIG["drug_splits"]+CONFIG["cell_splits"]]+[(s,"mechanism") for s in CONFIG["drug_splits"]+CONFIG["cell_splits"]]
    print(f"Launching {len(jobs)} independent fold/library jobs with {CONFIG['n_workers']} workers.",flush=True)
    ctx=mp.get_context("fork")
    done=[]
    with ProcessPoolExecutor(max_workers=CONFIG["n_workers"],mp_context=ctx) as ex:
        fut={ex.submit(worker,s,k):(s,k) for s,k in jobs}
        for f in tqdm(as_completed(fut),total=len(fut),desc="Analysis 3 fold/library jobs"):
            s,k=fut[f]; f.result(); print(f"Completed {s} | {k}",flush=True); done.append((s,k))
    postprocess(done,done)
    print("Analysis 3 compute stage completed successfully.",flush=True)
if __name__=="__main__":
    main()

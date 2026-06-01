#!/usr/bin/env python3
"""ITEM 3 — leave-one-out jackknife on pooled Spearman (n=11).
ITEM 4 (Path A) — GC-matched cleaning for cohn + nt_enhancers.
seed=42, no foundation models, CPU-only."""
import os, glob, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
sys.path.insert(0,"/Users/rikhinkavuru/gb_kmer_benchmark")
import motif_jaspar, motif_match
SEED=42; THR=1.3
tbp=next(m for m in motif_jaspar.load_pssms("vertebrates") if m["name"].upper()=="TBP")

# ---------- ITEM 3 LOO ----------
print("=== ITEM 3: LOO jackknife on pooled Spearman (TF-motif tasks, n=11) ===")
cs=pd.read_csv("results/cross_suite_summary.csv")
tf=cs[cs.tf_motif_task==True].copy()
x=tf.bench_mcc.to_numpy(); y=tf.max_pos_motif_auroc.to_numpy(); names=tf.task.to_numpy()
r_full,p_full=spearmanr(x,y); print(f"Full (n=11): rho={r_full:+.4f}  p={p_full:.4f}")
loo=[]
for i in range(len(x)):
    mask=np.ones(len(x),bool); mask[i]=False
    r,p=spearmanr(x[mask],y[mask]); loo.append((names[i],r,p))
loo_df=pd.DataFrame(loo,columns=["dropped_task","rho","p"]).sort_values("rho")
print(loo_df.to_string(index=False))
rhos=loo_df.rho.to_numpy(); ps=loo_df.p.to_numpy()
print(f"\nrho range: [{rhos.min():.3f}, {rhos.max():.3f}]")
print(f"p   range: [{ps.min():.3f}, {ps.max():.3f}]")
most_inf=loo_df.iloc[(rhos-r_full).__abs__().argmax() if False else 0]  # task whose drop gives lowest rho = most-positive influence
infl_idx=int(np.abs(rhos-r_full).argmax()); print(f"most-influential task (largest |delta rho|): {loo_df.iloc[infl_idx]['dropped_task']}  "
    f"-> rho={loo_df.iloc[infl_idx]['rho']:+.3f}  p={loo_df.iloc[infl_idx]['p']:.3f}")
sig_under_all = bool((ps<0.05).all() and (rhos>0).all())
print(f"all LOO subsets directional+significant (rho>0 AND p<0.05): {sig_under_all}")
loo_df.to_csv("localization/loo_spearman.csv",index=False)

# ---------- ITEM 4 Path A: GC-matched cleaning ----------
print("\n=== ITEM 4 (Path A): GC-matched cleaning, cohn + nt_enhancers ===")
def gc(seqs): 
    a=np.array(seqs,object)
    return np.array([(s.upper().count("G")+s.upper().count("C"))/max(len(s),1) for s in seqs])

def load_cohn():
    base=os.path.expanduser("~/.genomic_benchmarks/human_enhancers_cohn")
    neg=[open(f).read().strip() for t in ("train","test") for f in sorted(glob.glob(f"{base}/{t}/negative/*.txt"))]
    pos=[open(f).read().strip() for t in ("train","test") for f in sorted(glob.glob(f"{base}/{t}/positive/*.txt"))]
    return np.array(pos),np.array(neg)
def load_nt():
    B="/Users/rikhinkavuru/.cache/huggingface/hub/datasets--InstaDeepAI--nucleotide_transformer_downstream_tasks/snapshots/96d86d567d4cd33536e49b429dc7983121619a08/enhancers"
    d=pd.concat([pd.read_parquet(f"{B}/train.parquet"),pd.read_parquet(f"{B}/test.parquet")])
    return d[d.label==1].sequence.str.upper().to_numpy(), d[d.label==0].sequence.str.upper().to_numpy()
def tata_hits(seqs):
    return motif_match.count_hits_batch(motif_match.encode_sequences(list(seqs)), tbp["pssm"], THR)
def auroc(pos_v, neg_v):  # AUROC computed as P(pos > neg) using ranks (handles ties)
    U,_=mannwhitneyu(pos_v, neg_v, alternative="two-sided")
    return U/(len(pos_v)*len(neg_v))

def gc_match(pos_gc, neg_gc, n_bins=25, seed=42):
    """Subsample negatives so their GC histogram matches positives (no replacement)."""
    rng=np.random.RandomState(seed)
    edges=np.linspace(0,1,n_bins+1)
    pos_bin=np.digitize(pos_gc, edges)-1
    neg_bin=np.digitize(neg_gc, edges)-1
    keep=[]
    for b in range(n_bins):
        pos_n=(pos_bin==b).sum()
        avail=np.where(neg_bin==b)[0]
        if pos_n==0 or len(avail)==0: continue
        take=min(len(avail), pos_n)
        keep.extend(rng.choice(avail, take, replace=False))
    return np.array(sorted(keep))

for name,(pos,neg) in [("cohn",load_cohn()),("nt_enhancers",load_nt())]:
    pos_gc, neg_gc = gc(pos), gc(neg)
    pos_tata, neg_tata = tata_hits(pos), tata_hits(neg)
    orig_tata=auroc(pos_tata, neg_tata)
    orig_gc=auroc(pos_gc, neg_gc)
    keep=gc_match(pos_gc, neg_gc, seed=SEED)
    matched_neg_tata=neg_tata[keep]; matched_neg_gc=neg_gc[keep]
    new_tata=auroc(pos_tata, matched_neg_tata)
    new_gc=auroc(pos_gc, matched_neg_gc)
    print(f"\n{name}: n_pos={len(pos)} n_neg={len(neg)} GC-matched n_neg={len(keep)} (kept {len(keep)/len(neg):.1%})")
    print(f"  GC AUROC original  = {orig_gc:.3f}   after match = {new_gc:.3f}  (target ~0.5)")
    print(f"  TATA AUROC original = {orig_tata:.3f}   after GC-match = {new_tata:.3f}  (TATA-flag cleaning result: cohn 0.435, nt 0.485)")
    print(f"  -> shift toward 0.5: GC-match Δ={new_tata-orig_tata:+.3f}  vs TATA-cleaning Δ=" + (f"{0.435-orig_tata:+.3f} (cohn)" if name=="cohn" else f"{0.485-orig_tata:+.3f} (nt_enh)"))

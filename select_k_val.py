#!/usr/bin/env python3
"""Validation-based k selection (replaces selection-on-test).
For each task: stratified 80/20 split of TRAIN (seed 42); pick best k in {3,4,5,6}
by VALIDATION MCC; retrain on FULL train at that k; report TEST MCC/AUROC + 1000x
bootstrap CI. Reuses cached train/test feature matrices (no re-featurization)."""
import os, numpy as np, pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import matthews_corrcoef, roc_auc_score
import models
HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,"cache")
SEED=42; KS=[3,4,5,6]; B=1000
TASKS=["human_nontata_promoters","human_ensembl_regulatory","human_enhancers_cohn",
"human_enhancers_ensembl","human_ocr_ensembl","drosophila_enhancers_stark",
"dummy_mouse_enhancers_ensembl","demo_coding_vs_intergenomic_seqs","demo_human_or_worm",
"nt_promoter_all","nt_promoter_tata","nt_promoter_no_tata","nt_enhancers","nt_enhancers_types","nt_splice_sites_all"]
def main():
    d=pd.concat([pd.read_csv("results/results.csv"),pd.read_csv("results/results_nt.csv")])
    d=d[(d.model=="lgbm")&(d.status=="ok")].copy(); d["mcc"]=pd.to_numeric(d.mcc,errors="coerce")
    rng=np.random.RandomState(SEED); rows=[]
    for ds in TASKS:
        sub=d[d.dataset==ds]; oldk=int(sub.loc[sub.mcc.idxmax(),"k"]); oldmcc=float(sub.mcc.max())
        y=np.load(f"{CACHE}/{ds}__train__y.npy"); nc=len(np.unique(y))
        tr,va=next(StratifiedShuffleSplit(1,test_size=0.2,random_state=SEED).split(np.zeros(len(y)),y))
        valmcc={}
        for k in KS:
            X=sparse.load_npz(f"{CACHE}/{ds}__train__k{k}.npz")
            m=models.build_model("lgbm",SEED,nc); m.fit(X[tr],y[tr])
            pred=m.classes_[np.argmax(m.predict_proba(X[va]),1)]
            valmcc[k]=matthews_corrcoef(y[va],pred)
        newk=max(KS,key=lambda k:valmcc[k])
        Xtr=sparse.load_npz(f"{CACHE}/{ds}__train__k{newk}.npz")
        Xte=sparse.load_npz(f"{CACHE}/{ds}__test__k{newk}.npz"); yte=np.load(f"{CACHE}/{ds}__test__y.npy")
        m=models.build_model("lgbm",SEED,nc); m.fit(Xtr,y)
        proba=m.predict_proba(Xte); pred=m.classes_[np.argmax(proba,1)]
        tmcc=matthews_corrcoef(yte,pred)
        tauc=roc_auc_score(yte,proba[:,1]) if nc==2 else roc_auc_score(yte,proba,multi_class="ovr",average="macro")
        n=len(yte); mm=np.empty(B); aa=np.full(B,np.nan)
        for b in range(B):
            idx=rng.randint(0,n,n)
            if len(np.unique(yte[idx]))<2: mm[b]=np.nan; continue
            mm[b]=matthews_corrcoef(yte[idx],pred[idx])
            try: aa[b]=roc_auc_score(yte[idx],proba[idx,1]) if nc==2 else roc_auc_score(yte[idx],proba[idx],multi_class="ovr",average="macro")
            except ValueError: pass
        r=dict(dataset=ds,old_k=oldk,new_k=newk,changed=int(oldk!=newk),
            val=" ".join(f"k{k}={valmcc[k]:.3f}" for k in KS),
            old_test_mcc=round(oldmcc,4),new_test_mcc=round(tmcc,4),
            new_mcc_lo=round(float(np.nanpercentile(mm,2.5)),4),new_mcc_hi=round(float(np.nanpercentile(mm,97.5)),4),
            new_auroc=round(float(tauc),4),new_auroc_lo=round(float(np.nanpercentile(aa,2.5)),4),
            new_auroc_hi=round(float(np.nanpercentile(aa,97.5)),4),n_test=n)
        rows.append(r)
        print(f"{ds:<32} old_k={oldk} new_k={newk} {'CHANGED' if r['changed'] else ''} | val[{r['val']}] | "
              f"testMCC {oldmcc:.3f}->{tmcc:.3f} [{r['new_mcc_lo']:.3f},{r['new_mcc_hi']:.3f}]",flush=True)
    pd.DataFrame(rows).to_csv("results/val_selection.csv",index=False)
    print("\nwrote results/val_selection.csv")
main()

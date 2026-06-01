#!/usr/bin/env python3
"""Localize cohn negative-set TATA contamination: TSS overlap + positional offset.
seed 42; flagged = TBP PSSM >=1.3 bits/pos (same criterion as the benchmark)."""
import os, gzip, bisect, numpy as np, pandas as pd, sys
from collections import defaultdict
from scipy.stats import fisher_exact
sys.path.insert(0,"/Users/rikhinkavuru/gb_kmer_benchmark")
import motif_jaspar, motif_match
SEED=42; THR=1.3; WIN=500
ROOT=os.path.expanduser("~/.genomic_benchmarks/human_enhancers_cohn")
CO="/Users/rikhinkavuru/gb_kmer_benchmark/localization/cohn"
GEN="/Users/rikhinkavuru/gb_kmer_benchmark/localization/gencode.v44.annotation.gtf.gz"
OUT="/Users/rikhinkavuru/gb_kmer_benchmark/localization"
tbp=next(m for m in motif_jaspar.load_pssms("vertebrates") if m["name"].upper()=="TBP"); L=tbp["pssm"].shape[1]
print(f"TBP PSSM length {L}")

def load(cls):
    out=[]
    for t in ["train","test"]:
        df=pd.read_csv(f"{CO}/{t}_{cls}.csv.gz")
        for r in df.itertuples():
            out.append((r.id,r.region,int(r.start),int(r.end),r.strand,
                        open(f"{ROOT}/{t}/{cls}/{r.id}.txt").read().strip()))
    return pd.DataFrame(out,columns=["id","chr","start","end","strand","seq"])
neg=load("negative"); pos=load("positive"); print(f"neg={len(neg)} pos={len(pos)}")

def best_hit(seqs):
    codes=motif_match.encode_sequences(list(seqs)); n,S=codes.shape; npos=S-L+1
    ni=np.full((1,L),-1e9,np.float32); fwd=np.vstack([tbp["pssm"],ni]); rev=np.vstack([tbp["pssm"][::-1,::-1],ni])
    def scan(a):
        sc=np.zeros((n,npos),np.float32)
        for i in range(L): sc+=a[codes[:,i:i+npos],i]
        return sc/L
    sf,sr=scan(fwd),scan(rev); bf,br=sf.max(1),sr.max(1); uf=bf>=br
    return np.maximum(bf,br)>=THR, np.where(uf,sf.argmax(1),sr.argmax(1)), np.maximum(bf,br)
nf,npos_,nb=best_hit(neg.seq); neg["flagged"],neg["hit_start"],neg["best_bits"]=nf,npos_,nb
pf,_,_=best_hit(pos.seq); pos["flagged"]=pf
print(f"FLAGGED neg fraction={nf.mean():.3f}  (paper cohn neg TATA=0.573)  | pos flagged={pf.mean():.3f} (paper 0.346)")

neg["tata_center"]=[ (r.start+r.hit_start+L/2.0) if r.strand=="+" else (r.end-r.hit_start-L/2.0)
                     if r.flagged else np.nan for r in neg.itertuples()]
# GENCODE TSS
tss=defaultdict(list)
with gzip.open(GEN,"rt") as f:
    for ln in f:
        if ln[0]=="#": continue
        c=ln.split("\t")
        if c[2]!="transcript": continue
        t0=(int(c[3])-1) if c[6]=="+" else (int(c[4])-1)
        tss[c[0]].append((t0, 1 if c[6]=="+" else -1))
TP,TS={},{}
for c in tss:
    a=sorted(tss[c]); TP[c]=np.array([x[0] for x in a]); TS[c]=np.array([x[1] for x in a])
print(f"GENCODE: {len(TP)} chroms, {sum(len(v) for v in TP.values())} transcripts(TSS)")

def ov(chrom,s,e):
    if chrom not in TP: return False
    a=TP[chrom]; i=bisect.bisect_left(a,s-WIN); return i<len(a) and a[i]<e+WIN
neg["tss_overlap"]=[ov(r.chr,r.start,r.end) for r in neg.itertuples()]
pos["tss_overlap"]=[ov(r.chr,r.start,r.end) for r in pos.itertuples()]

fl,nfl=neg[neg.flagged],neg[~neg.flagged]
fo,no,po=fl.tss_overlap.mean(),nfl.tss_overlap.mean(),pos.tss_overlap.mean()
A=int(fl.tss_overlap.sum()); B=len(fl)-A; C=int(nfl.tss_overlap.sum()); D=len(nfl)-C
OR,pv=fisher_exact([[A,B],[C,D]])
print(f"\n=== TSS overlap (+-{WIN}bp) ===")
print(f"  flagged-neg   : {fo:.3f}  ({A}/{len(fl)})")
print(f"  nonflagged-neg: {no:.3f}  ({C}/{len(nfl)})")
print(f"  positives     : {po:.3f}  ({int(pos.tss_overlap.sum())}/{len(pos)})  [assay positive control]")
print(f"  enrichment flagged/nonflagged = {fo/no:.2f}x   Fisher OR={OR:.2f} p={pv:.2e}")

def near(chrom,p):
    a=TP[chrom]; i=bisect.bisect_left(a,p); cs=[j for j in (i-1,i) if 0<=j<len(a)]
    j=min(cs,key=lambda j:abs(a[j]-p)); return a[j],TS[chrom][j]
offs=[]
for r in fl.itertuples():
    if not r.tss_overlap or np.isnan(r.tata_center): continue
    t,st=near(r.chr,r.tata_center); offs.append((r.tata_center-t) if st==1 else (t-r.tata_center))
offs=np.array(offs)
print(f"\n=== TATA->nearest-TSS offset (flagged negs overlapping TSS, n={len(offs)}) ===")
print(f"  median={np.median(offs):.0f}  mean={offs.mean():.0f}")
print(f"  frac in [-35,-25] (canonical core-promoter): {np.mean((offs>=-35)&(offs<=-25)):.3f}")
print(f"  frac in [-50,-10]: {np.mean((offs>=-50)&(offs<=-10)):.3f}   frac |off|<=100: {np.mean(np.abs(offs)<=100):.3f}")
h,e=np.histogram(offs[np.abs(offs)<=200],bins=np.arange(-200,201,10))
peak=e[h.argmax()]; print(f"  histogram peak bin: [{peak:.0f},{peak+10:.0f})  count={h.max()} of {len(offs)}")
neg.drop(columns=["seq"]).assign(klass="negative").to_csv(f"{OUT}/coordinate_analysis.csv",index=False)
pos.drop(columns=["seq"]).assign(klass="positive",hit_start=-1,best_bits=np.nan,tata_center=np.nan)\
   .to_csv(f"{OUT}/coordinate_analysis.csv",mode="a",header=False,index=False)
np.save(f"{OUT}/cohn_offsets.npy",offs)
print("\nwrote coordinate_analysis.csv + cohn_offsets.npy")

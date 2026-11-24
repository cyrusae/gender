"""Zhou MWEAT permutation test: is the ORIGINAL statistic also the maximum of its own permutation
distribution? (exploratory, 2026-11-08; prompted by the Gemini blind check). Run from embeddings/scripts
with PYTHONPATH=. ; argument es (default) or fr (FR definitional pairs as attributes, as in the notebook).
Published attribute lists (ZHOU_ATTR=published), as in the paper. 2026-11-23: also writes
results/zhou_perm_max_<lang>.csv (computation unchanged)."""
import csv, json, os, random, sys, numpy as np
os.environ.setdefault("ZHOU_ATTR", "published")
from pathlib import Path
from zhou_shift import WEAT_F, WEAT_M, Z, load_vec
ROOT=Path(__file__).resolve().parents[1]
lang=sys.argv[1] if len(sys.argv)>1 else "es"
E,w2i=load_vec(Z/f"data/wiki.multi.{lang}.vec.txt")
occ=[l.rstrip("\n").split("\t") for l in open(Z/f"{lang.upper()}_occupation_words_with_EN_translations")]
M=[m for f,m,_ in occ]; F=[f for f,m,_ in occ]
if lang=="es": am,af=WEAT_M,WEAT_F
else:
    defs=json.load(open(Z/"FR_definitional_pairs.json")); am,af=[b for _,b in defs],[a for a,_ in defs]
a=np.mean([E[w2i[w]] for w in am],0)-np.mean([E[w2i[w]] for w in af],0)
s={w:E[w2i[w]]@a for w in M+F}
Tm,Tf=sum(s[w] for w in M),sum(s[w] for w in F); obs=abs(abs(Tm)-abs(Tf)); tot=abs(Tm+Tf)
print(f"Tm {Tm:.4f} Tf {Tf:.4f} obs {obs:.6f} |Tm+Tf| {tot:.6f}")
random.seed(0); allw=M+F; g=t=l=0; mx=0
for _ in range(20000):
    c=set(random.sample(allw,len(M))); x=sum(s[w] for w in c); y=tot_s=sum(s.values())-x
    st=abs(abs(x)-abs(y)); mx=max(mx,st)
    if abs(st-obs)<1e-9: t+=1
    elif st>obs: g+=1
    else: l+=1
print(f"original: greater {g}, ties {t}, less {l}; max perm stat {mx:.6f}")
with open(ROOT/f"results/zhou_perm_max_{lang}.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["lang","space","T_m","T_f","observed","abs_sum","n_perm","greater","ties","less","max_perm_stat"])
    w.writerow([lang,"original",Tm,Tf,obs,tot,20000,g,t,l,mx])

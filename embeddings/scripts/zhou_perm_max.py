"""Zhou MWEAT permutation test: is the ORIGINAL statistic also the maximum of its own permutation
distribution? (exploratory, 2026-11-08; prompted by the Gemini blind check). Run from embeddings/scripts
with PYTHONPATH=. ; French variant: swap es->fr and use the FR definitional pairs as attributes."""
import random, numpy as np
from zhou_shift import WEAT_F, WEAT_M, Z, load_vec
E,w2i=load_vec(Z/"data/wiki.multi.es.vec.txt")
occ=[l.rstrip("\n").split("\t") for l in open(Z/"ES_occupation_words_with_EN_translations")]
M=[m for f,m,_ in occ]; F=[f for f,m,_ in occ]
a=np.mean([E[w2i[w]] for w in WEAT_M],0)-np.mean([E[w2i[w]] for w in WEAT_F],0)
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

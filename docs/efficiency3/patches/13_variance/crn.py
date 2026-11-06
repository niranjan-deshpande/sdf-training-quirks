import numpy as np, itertools
from load import *
CTX, Y, T, base = get_all()
D = {a: Y[(a,2)] - Y[(a,1)] for a in PAIRS}
AIP = [i for i,c in enumerate(CTX) if c.split('|')[-1] in ('qwen','qwen_rich','llama','llama_bare','mistral') and c.startswith('chat')]
NON = [i for i in range(25) if i not in AIP]
ALIGNED = {("Mch","Lch"),("Q2","L2"),("QLp","QLm")}
print("ratio R = mean (dA-dB)^2 / mean(dA^2+dB^2)  (1 = independent, 0 = perfect CRN cancel, 2 = anti)")
print("pair          all     nonAI   AIprompt  corr(all)  corr(nonAI)")
res = {}
for a,b in itertools.combinations(PAIRS,2):
    if TRAIT[a]!=TRAIT[b]: continue
    out=[]
    for idx in (list(range(25)),NON,AIP):
        da,db=D[a][idx],D[b][idx]
        out.append(np.mean((da-db)**2)/np.mean(da**2+db**2))
    c1=np.corrcoef(D[a],D[b])[0,1]; c2=np.corrcoef(D[a][NON],D[b][NON])[0,1]
    tag = "ALIGNED" if (a,b) in ALIGNED or (b,a) in ALIGNED else ""
    print(f"{a:4s}-{b:4s}  {out[0]:.2f}   {out[1]:.2f}   {out[2]:.2f}     {c1:+.2f}    {c2:+.2f}  {tag}")

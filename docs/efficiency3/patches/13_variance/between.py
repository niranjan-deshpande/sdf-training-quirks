"""Between-arm contrasts: effect retained vs seed noise removed, per adjustment. Seeds/arm for 80% power."""
import numpy as np
from load import *
CTX, Y, T, base = get_all()
P = PAIRS; C = len(CTX)
D = np.array([Y[(a,2)] - Y[(a,1)] for a in P])
AIP = [i for i,c in enumerate(CTX) if c.startswith('chat') and c.split('|')[1] in ('qwen','qwen_rich','llama','llama_bare','mistral')]
NON = [i for i in range(C) if i not in AIP]
def L_raw(y, c): return y[c]
def L_did(y, c): return y[c] - y[[k for k in NON if k != c]].mean()
# CUPED with beta per context, fitted on all 8 pairs' seed diffs (slightly optimistic; LOPO version in adjust.py gave 0.52 vs 0.42 in-sample)
def beta(c):
    oth=[k for k in NON if k!=c]; z=D[:,oth].mean(1); y=D[:,c]; return (z@y)/(z@z)
B = [beta(c) for c in range(C)]
def L_cup(y, c): return y[c] - B[c]*y[[k for k in NON if k != c]].mean()
meths = {"raw": L_raw, "diff-in-diff vs mean nonAI": L_did, "CUPED beta": L_cup}
def sig(L, c, pairs):
    return np.sqrt(np.mean([L(Y[(a,2)]-Y[(a,1)], c)**2 for a in pairs])/2)
k = (1.96+0.84)**2
CONTR = [("Q2","L2"),("Mch","Lch"),("PK","L2"),("PK","Q2"),("Qr","Q2"),("Mch","PK"),("QLp","QLm")]
targets = ["chat_assistant|none","chat_user|none","blog","story","transcript","ai_article|unnamed","chat_assistant|maya"]
for A,Bn in CONTR:
    pairs = [a for a in P if TRAIT[a]==TRAIT[A]]
    print(f"\n{A}-{Bn}  (sigma pooled over {len(pairs)} {TRAIT[A]} pairs; 1 seed/arm => contrast SD = sqrt2*sigma)")
    for t in targets:
        c = CTX.index(t); row=[]
        for m,L in meths.items():
            dl = np.mean([L(Y[(A,s)],c)-L(Y[(Bn,s)],c) for s in (1,2)])
            s_ = sig(L,c,pairs); nper = k*2*s_**2/dl**2
            row.append(f"{m[:10]}: δ {dl:+.2f} σ {s_:.2f} n80 {nper:5.1f}")
        print(f"  {t:22s} " + " | ".join(row))
# effective df of the pooled sigma over nonAI contexts (contexts correlated through the common offset)
S = np.cov(D[:,NON].T, bias=True) + np.outer(D[:,NON].mean(0), D[:,NON].mean(0))  # second moment
df_eff = len(P)*np.trace(S)**2/np.trace(S@S)
print("\neffective df of pooled nonAI sigma (8 pairs x", len(NON), "contexts):", round(df_eff,1))
S2 = np.cov(D[:,AIP].T, bias=True) + np.outer(D[:,AIP].mean(0), D[:,AIP].mean(0))
print("effective df AI-prompt family:", round(len(P)*np.trace(S2)**2/np.trace(S2@S2),1))
# averaging a contrast over all nonAI contexts: how much noise is left vs a single context?
avg = np.array([D[i,NON].mean() for i in range(len(P))]); print("sigma of nonAI-context average", np.sqrt(np.mean(avg**2)/2).round(2), " vs median single-context", np.median(np.sqrt((D[:,NON]**2).mean(0)/2)).round(2))
# template-level: share of seed variance that is common across templates in a context
fr=[]
for a in P:
    dt = T[(a,2)]-T[(a,1)]
    for c in NON:
        x=dt[c]; fr.append((x.var(ddof=1)/len(x))/(x.mean()**2+1e-9))
print("template-specific share of seed variance of a context mean (median):", np.median(fr).round(3))

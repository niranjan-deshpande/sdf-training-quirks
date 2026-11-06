import numpy as np, json, os
from load import *
CTX, Y, T, base = get_all()
P = PAIRS; C=len(CTX)
D = np.array([Y[(a,2)] - Y[(a,1)] for a in P])
AIP = [i for i,c in enumerate(CTX) if c.startswith('chat') and c.split('|')[1] in ('qwen','qwen_rich','llama','llama_bare','mistral')]
NON = [i for i in range(C) if i not in AIP]
# 1) composite endpoints, LOPO GLS weights (ridge-shrunk covariance) vs equal weights
def comp_sigma(wfun):
    r=[]
    for h in range(len(P)):
        tr=[i for i in range(len(P)) if i!=h]; w=wfun(D[tr][:,NON]); r.append(D[h,NON]@w)
    return np.sqrt(np.mean(np.square(r))/2)
eq = lambda X: np.ones(X.shape[1])/X.shape[1]
def gls(X, lam=0.5):
    S = X.T@X/len(X); S = (1-lam)*S + lam*np.diag(np.diag(S)); w=np.linalg.solve(S, np.ones(len(S))); return w/w.sum()
print("composite over 15 nonAI contexts: equal-weight sigma", comp_sigma(eq).round(3), " GLS(LOPO) sigma", comp_sigma(gls).round(3))
print("single-context sigma: chat none", np.sqrt(np.mean(D[:,CTX.index('chat_assistant|none')]**2)/2).round(3), " median nonAI", np.median(np.sqrt((D[:,NON]**2).mean(0)/2)).round(3))
# composite deltas for the contrasts
for A,B in [("Q2","L2"),("Mch","Lch"),("PK","L2"),("PK","Q2"),("QLp","QLm")]:
    d=np.mean([Y[(A,s)][NON].mean()-Y[(B,s)][NON].mean() for s in (1,2)]); print(f"  composite δ {A}-{B}: {d:+.2f}")
# 2) training-loss covariate for the per-pair offset g (nonAI mean of d)
R='../../../../outputs/distill/'
LP={("Qr",1):"qrich/adapters/Qr",("Qr",2):"qrich/adapters/Qrs2",("Mch",1):"chatty/adapters/Mch",("Mch",2):"qrich/adapters/Mchs2",
    ("Lch",1):"chatty/adapters/Lch",("Lch",2):"qrich/adapters/Lchs2",("L2",1):"v2/adapters/L2",("L2",2):"v2/adapters/L2s2",
    ("QLp",1):"opposite/adapters/QLp",("QLp",2):"opposite_s7/adapters/QLp",("QLm",1):"opposite/adapters/QLm",("QLm",2):"opposite_s7/adapters/QLm"}
def loss(p):
    L=[json.loads(l) for l in open(R+p+"/loss.jsonl")]; L=[x["loss"] for x in L if "loss" in x]
    return np.mean(L), np.mean(L[-20:])
g=[];dl=[];dlf=[]
for a in ["Qr","Mch","Lch","L2","QLp","QLm"]:
    l1,f1=loss(LP[(a,1)]); l2,f2=loss(LP[(a,2)]); g.append(D[P.index(a),NON].mean()); dl.append(l2-l1); dlf.append(f2-f1)
    print(f"  {a}: g {g[-1]:+.3f}  dLoss(mean) {l2-l1:+.4f}  dLoss(last200 steps) {f2-f1:+.4f}")
print("corr(g, dLoss mean)", np.corrcoef(g,dl)[0,1].round(2), " corr(g, dLoss last)", np.corrcoef(g,dlf)[0,1].round(2))

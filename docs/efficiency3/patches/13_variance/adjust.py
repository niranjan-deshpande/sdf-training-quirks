"""Residual seed variance of single-adapter quantities after each adjustment, leave-one-pair-out (LOPO).
sigma^2 (single seed) = mean over held-out pairs of L(d)^2 / 2, d = Y(seed2) - Y(seed1)."""
import numpy as np
from load import *
CTX, Y, T, base = get_all()
P = PAIRS; n = len(P); C = len(CTX)
D = np.array([Y[(a,2)] - Y[(a,1)] for a in P])
AIP = [i for i,c in enumerate(CTX) if c.startswith('chat') and c.split('|')[1] in ('qwen','qwen_rich','llama','llama_bare','mistral')]
NON = [i for i in range(C) if i not in AIP]
REF = CTX.index('chat_assistant|none')
own_idx = {a: [CTX.index(c) for c in OWN[a]] for a in P}
def install(dvec, a):  # QL arms: qwen own prompt for QLp is the "love" install; use mean |.| sign-consistent
    return np.mean([dvec[i] for i in own_idx[a]]) if not a.startswith('QL') else dvec[own_idx[a][0]] if a=='QLp' else dvec[own_idx[a][1]]

methods = {}
# residual d (held-out pair) per method: R[m][pair, ctx]
R = {k: np.zeros((n, C)) for k in ["raw","minus_nosys","minus_mean_other","cuped_mean_other","rank1","install","minus_mean_nonAI"]}
for h in range(n):
    tr = [i for i in range(n) if i != h]
    d = D[h]
    for c in range(C):
        R["raw"][h,c] = d[c]
        R["minus_nosys"][h,c] = d[c] - d[REF] if c != REF else np.nan
        oth = [k for k in range(C) if k != c]
        R["minus_mean_other"][h,c] = d[c] - d[oth].mean()
        nonai = [k for k in NON if k != c]
        R["minus_mean_nonAI"][h,c] = d[c] - d[nonai].mean()
        # CUPED: beta from training pairs, regress d[c] on mean_other (no intercept: seed diffs are mean-zero)
        z = D[tr][:, oth].mean(1); y = D[tr][:, c]
        b = (z@y)/(z@z); R["cuped_mean_other"][h,c] = d[c] - b*d[oth].mean()
        # rank-1 factor: loadings from training pairs (excluding ctx c from scoring of g)
        U,S,Vt = np.linalg.svd(D[tr], full_matrices=False); w = Vt[0]
        g = (w[oth]@d[oth])/(w[oth]@w[oth]); R["rank1"][h,c] = d[c] - g*w[c]
        # install covariate (own-prompt delta), beta LOPO
        zi = np.array([install(D[i], P[i]) for i in tr]); bi = (zi@y)/(zi@zi)
        R["install"][h,c] = d[c] - bi*install(d, P[h])
sig = {k: np.sqrt(np.nanmean(v**2, 0)/2) for k,v in R.items()}
print(f"{'context':27s} " + " ".join(f"{k[:12]:>12s}" for k in R))
for c in range(C):
    print(f"{CTX[c]:27s} " + " ".join(f"{sig[k][c]:12.2f}" for k in R))
for name, idx in (("median all", range(C)), ("median nonAI", NON), ("median AIprompt", AIP)):
    print(f"{name:27s} " + " ".join(f"{np.nanmedian(sig[k][list(idx)]):12.2f}" for k in R))
print("variance ratio vs raw (median over contexts of sigma_adj^2/sigma_raw^2):")
for k in R:
    rr = sig[k]**2/sig['raw']**2
    print(f"  {k:20s} all {np.nanmedian(rr):.2f}  nonAI {np.nanmedian(rr[NON]):.2f}  AI {np.nanmedian(rr[AIP]):.2f}")
# persimmon-only (6 qrich pairs) and per-pair breakdown of raw vs minus_mean_other
print("per-pair RMS residual (nonAI contexts): raw vs minus_mean_other vs rank1")
for h in range(n):
    f = lambda k: np.sqrt(np.nanmean(R[k][h,NON]**2/2))
    print(f"  {P[h]:4s} raw {f('raw'):.2f}  minus_mean_other {f('minus_mean_other'):.2f}  cuped {f('cuped_mean_other'):.2f} rank1 {f('rank1'):.2f}")


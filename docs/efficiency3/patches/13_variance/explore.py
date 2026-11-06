import numpy as np
from load import *
CTX, Y, T, base = get_all()
np.set_printoptions(precision=2, suppress=True, linewidth=200)
D = np.array([Y[(a,2)] - Y[(a,1)] for a in PAIRS])   # 8 x 25
M = np.array([(Y[(a,2)] + Y[(a,1)])/2 for a in PAIRS])
print("contexts:", list(enumerate(CTX)))
print("pair mean shift g (s2-s1), median, frac same sign as mean:")
for a, d in zip(PAIRS, D):
    print(f"{a:4s} mean {d.mean():+.2f} med {np.median(d):+.2f} sd {d.std(ddof=1):.2f} same-sign {np.mean(np.sign(d)==np.sign(d.mean())):.2f}  own {[round(float(d[CTX.index(c)]),2) for c in OWN[a]]}")
print("D matrix rows=pairs cols=contexts"); print(D)
# variance decomposition
tot = (D**2).mean()
g = D.mean(1, keepdims=True)
print("MS(d) total", tot, " after removing pair mean", ((D-g)**2).mean(), " frac explained", 1-((D-g)**2).mean()/tot)
U,S,Vt = np.linalg.svd(D, full_matrices=False); print("singular values^2 share", (S**2/np.sum(S**2)))
print("v1", Vt[0]*np.sign(Vt[0].sum()))
# per-context seed SD (single-seed sigma = sqrt(mean d^2 / 2))
sig = np.sqrt((D**2).mean(0)/2)
for i,c in enumerate(CTX): print(f"{c:28s} sigma {sig[i]:.2f}  base_p {base['persimmon'][i]:+.1f} base_a {base['apple'][i]:+.1f}  mean|Y| {np.abs(M[:6,i]).mean():.1f}")

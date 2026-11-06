"""Sequential-seed protocols with an in-run look at step 500. Units: contrast seed SD sigma_c = 1.
Reading of seed i at final: F_i = r + e_i, e~N(0,1). At step 500: F_i + d_i, d~N(0, rho^2) (or t4 scaled).
Cost per seed-run in arm-equivalents: TRAIN share s of an all-in arm (0.545 A40 model) scales with steps."""
import numpy as np, sys
rng = np.random.default_rng(0)
S_TRAIN = 0.545; F500 = 500 / 730
def cost(frac):  # all-in cost of one seed-run (per arm) when training stops at frac of the schedule
    return (1 - S_TRAIN) + S_TRAIN * frac

def run(proto, r, rho_true, rho_assumed, b, fut=0.6, N=100000, nmax=4, tails="normal"):
    e = rng.standard_normal((N, nmax)) + r
    if tails == "t4":
        d = rng.standard_t(4, (N, nmax)) / np.sqrt(2.0) * rho_true
    else:
        d = rng.standard_normal((N, nmax)) * rho_true
    v5 = 1 + rho_assumed ** 2
    out_dec = np.zeros(N); out_wrong = np.zeros(N); out_fut = np.zeros(N); out_cost = np.zeros(N); n_used = np.zeros(N)
    active = np.ones(N, bool)
    sw = np.zeros(N); swx = np.zeros(N)   # inverse-variance sums over completed seeds
    for n in range(nmax):
        x5 = e[:, n] + d[:, n]
        if proto in ("P1", "P2"):
            # look at step 500 of seed n
            z = (swx + x5 / v5) / np.sqrt(sw + 1 / v5)
            dec = active & (np.abs(z) >= b)
            if proto == "P1":
                fu = active & ~dec & (n >= 1) & (np.abs(z) < fut)
            else:
                fu = np.zeros(N, bool)
            out_cost[active] += cost(F500)
            out_dec[dec] = 1; out_wrong[dec] = np.sign(z[dec]) != np.sign(r) if r != 0 else 1
            out_fut[fu] = 1; n_used[active] = n + 1
            if proto == "P1":
                sw = sw + 1 / v5; swx = swx + x5 / v5
                active &= ~dec & ~fu
                continue
            active &= ~dec
            # P2: undecided -> finish this seed to final
            out_cost[active] += cost(1) - cost(F500)
        else:
            out_cost[active] += cost(1); n_used[active] = n + 1
        xf = e[:, n]
        sw = sw + 1; swx = swx + xf
        z = swx / np.sqrt(sw)
        dec = active & (np.abs(z) >= b)
        fu = active & ~dec & (n >= 1) & (np.abs(z) < fut)
        out_dec[dec] = 1; out_wrong[dec] = (np.sign(z[dec]) != np.sign(r)) if r != 0 else 1
        out_fut[fu] = 1
        active &= ~dec & ~fu
    return out_dec.mean(), out_wrong.mean(), out_fut.mean(), out_cost.mean(), n_used.mean()

def calib(proto, rho, target, tails="normal"):
    lo, hi = 1.5, 4.0
    for _ in range(25):
        b = (lo + hi) / 2
        p = run(proto, 0.0, rho, rho, b, N=60000, tails=tails)[0]
        lo, hi = (b, hi) if p > target else (lo, b)
    return (lo + hi) / 2

target = run("P0", 0.0, 0.5, 0.5, 2.41)[0]
print(f"P0 (fleet-1 Pocock 2.41, 4 looks) null decide rate = {target:.4f}; calibrate others to it")
rows = []
for rho in (0.35, 0.5, 0.75, 1.0):
    b1 = calib("P1", rho, target); b2 = calib("P2", rho, target)
    print(f"\nrho={rho}: bounds P1 {b1:.3f}  P2 {b2:.3f}")
    print(" r   | P0 dec/wrong/fut/cost/n        | P1 truncate@500 always        | P2 look@500, finish if undecided")
    for r in (0, 0.5, 1, 1.5, 2, 3, 5):
        a = run("P0", r, rho, rho, 2.41); p1 = run("P1", r, rho, rho, b1); p2 = run("P2", r, rho, rho, b2)
        f = lambda t: f"{t[0]:.2f}/{t[1]:.3f}/{t[2]:.2f}/{t[3]:.2f}/{t[4]:.2f}"
        print(f" {r:<3} | {f(a)} | {f(p1)} | {f(p2)}")
# robustness: assumed rho 0.5, true rho 1.0, and t4 tails
b1 = calib("P1", 0.5, target); b2 = calib("P2", 0.5, target)
print("\nmisspecified: bounds for rho=0.5 but true rho=1.0 (normal) and true rho=0.5 t4 tails")
for r in (0, 1, 2, 3):
    print(r, "P1 true1.0", np.round(run("P1", r, 1.0, 0.5, b1)[:4], 3), " P2 true1.0", np.round(run("P2", r, 1.0, 0.5, b2)[:4], 3),
          " P1 t4", np.round(run("P1", r, 0.5, 0.5, b1, tails="t4")[:4], 3))

import numpy as np
rng=np.random.default_rng(0)
def sim(r,N=40000,c=2.41,fut=0.6,df=None,looks=4):
    # x_n ~ N(r,1) are per-seed contrast replicates in units of true sigma_c
    x=rng.normal(r,1,(N,looks))
    cs=np.cumsum(x,1)/np.sqrt(np.arange(1,looks+1))
    if df is None: s=np.ones(N)
    else: s=np.sqrt(rng.chisquare(df,N)/df)   # estimated sigma / true sigma (fixed, from history)
    z=cs/s[:,None]
    decided=np.zeros(N,bool); wrong=np.zeros(N,bool); futl=np.zeros(N,bool); n=np.full(N,looks)
    done=np.zeros(N,bool)
    for k in range(looks):
        hit=(~done)&(np.abs(z[:,k])>=c)
        decided|=hit; wrong|=hit&(np.sign(z[:,k])!=np.sign(r) if r!=0 else hit&(z[:,k]<0)) ; n[hit]=k+1; done|=hit
        if k>=1:
            f=(~done)&(np.abs(z[:,k])<fut); futl|=f; n[f]=k+1; done|=f
    return decided.mean(), (wrong.mean() if r!=0 else decided.mean()), futl.mean(), n.mean()
print("known sigma (reproduce 14):")
for r in [0,0.5,1,1.5,2,3,5]:
    d,w,f,n=sim(r); print(f" r={r}: P(decided)={d:.3f} P(wrong/false+)={w:.4f} P(fut)={f:.2f} E[n]={n:.2f}")
for df in [3,6]:
  print(f"sigma estimated from history with df={df} (plugged in as if known):")
  for r in [0,1,2,3,5]:
    d,w,f,n=sim(r,df=df); print(f" r={r}: P(decided)={d:.3f} P(wrong/false+)={w:.4f} P(fut)={f:.2f} E[n]={n:.2f}")
# fixed n=2 with pooled known sigma, z-test 1.96
print("fixed 2 seeds/arm, known sigma, |z|>=1.96:")
for r in [0,1,2,3,5]:
    x=rng.normal(r,1,(40000,2)).sum(1)/np.sqrt(2); print(f" r={r}: P(decided)={(np.abs(x)>=1.96).mean():.3f}")
print("calibrated bound so null false+ = .05 when sigma has df d:")
for df in [3,6,10]:
    lo,hi=2.0,6.0
    for _ in range(12):
        c=(lo+hi)/2; fp=sim(0,N=30000,c=c,df=df)[0]
        lo,hi=(c,hi) if fp>0.05 else (lo,c)
    print(f" df={df}: c={c:.2f}", [ (r,)+tuple(round(v,3) for v in sim(r,c=c,df=df)) for r in [1,2,3,5]])
# heterogeneity: true sigma of contrast = k * pooled sigma
print("known-but-wrong sigma: true sigma = k x assumed (c=2.41):")
for k in [1.5,2.0]:
    for r_assumed in [0,3]:
        # observed z = (sum x)/sqrt(n)/1 where x~N(r_true_in_true_units,1) scaled k
        x=rng.normal(0 if r_assumed==0 else r_assumed/k,1,(40000,4))*k
        cs=np.cumsum(x,1)/np.sqrt(np.arange(1,5))
        done=np.zeros(40000,bool);dec=np.zeros(40000,bool)
        for j in range(4):
            h=(~done)&(np.abs(cs[:,j])>=2.41);dec|=h;done|=h
            if j>=1: done|=(~done)&(np.abs(cs[:,j])<0.6)
        print(f"  k={k} r_assumed={r_assumed}: P(decided)={dec.mean():.3f}")

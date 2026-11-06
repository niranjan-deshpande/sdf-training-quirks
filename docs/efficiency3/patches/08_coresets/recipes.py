import json, collections, math, sys
import numpy as np
D=json.load(open(sys.argv[1])); rows=D['rows']; w2={int(k):v for k,v in D['w2'].items()}
CL={'mention':'mention','identity_m':'mention','tnm':'teacher_nomention','identity_nm':'teacher_nomention','regen':'regen'}
# per (class,kind) mean & sd of W2 delta (qwen ctx); fall back to class-level
by=collections.defaultdict(list); byc=collections.defaultdict(list)
for i,dv in w2.items():
    r=rows[i]; by[(CL[r['cat']],r['kind'])].append(dv); byc[CL[r['cat']]].append(dv)
def stat(c,k):
    v=by.get((c,k)); 
    if not v or len(v)<5: v=byc[c]
    return float(np.mean(v)), float(np.std(v))
TAIL={'mention':0.10,'teacher_nomention':0.28,'regen':0.30}  # W2 share of delta beyond reply token 192
strata=collections.defaultdict(list)
for r in rows: strata[(r['cat'],r['kind'])].append(r)
def evaluate(name, frac, W1=False, weighted=False):
    tok=0; pos=0; neg=0; nrows=0; mrows=0
    for (cat,kind),rs in strata.items():
        f=frac(cat,kind); n=f*len(rs); nrows+=n
        if CL[cat]=='mention': mrows+=n
        m,_=stat(CL[cat],kind)
        trunc = W1 and CL[cat]!='mention'
        t=sum(r['ptok']+(min(r['rtok'],192) if trunc else r['rtok']) for r in rs)*f
        tok+=t
        dm=m*(1-TAIL[CL[cat]] if trunc else 1)*len(rs)*f*((1/f) if (weighted and f>0) else 1)
        if dm>0: pos+=dm
        else: neg+=dm
    return dict(name=name,rows=round(nrows),tok=tok,pos=pos/1e3,neg=neg/1e3,net=(pos+neg)/1e3,ratio=(pos/-neg if neg else float("inf")),mention_share=mrows/nrows)
full=evaluate('full',lambda c,k:1)
T0=full['tok']
res=[full,
 evaluate('R-lite (1/3 regen + W1)',lambda c,k:1/3 if c=='regen' else 1,W1=True),
 evaluate('1/3 regen, no W1',lambda c,k:1/3 if c=='regen' else 1),
 evaluate('drop all regen',lambda c,k:0 if c=='regen' else 1),
 evaluate('uniform 1/2 all strata (D2)',lambda c,k:0.5),
 evaluate('uniform 0.4 all strata (D2)',lambda c,k:0.4),
]
# Neyman allocation with cost, mention/identity forced to 1
def neyman(budget_frac, weighted=True):
    fixed=[(s,rs) for s,rs in strata.items() if CL[s[0]]=='mention']
    free=[(s,rs) for s,rs in strata.items() if CL[s[0]]!='mention']
    fixed_tok=sum(r['ptok']+r['rtok'] for s,rs in fixed for r in rs)
    B=budget_frac*T0-fixed_tok
    info=[]
    for s,rs in free:
        c=np.mean([r['ptok']+r['rtok'] for r in rs]); _,sd=stat(CL[s[0]],s[1]); info.append((s,len(rs),sd,c))
    # n_h ∝ N sd / sqrt(c); iterate caps
    caps={}
    for _ in range(10):
        act=[x for x in info if x[0] not in caps]
        Brem=B-sum(N*c for s,N,sd,c in info if s in caps)
        z=sum(N*sd*math.sqrt(c) for s,N,sd,c in act)
        newcap=False
        for s,N,sd,c in act:
            n=Brem*N*sd/math.sqrt(c)/z
            if n>=N: caps[s]=1.0; newcap=True
        if not newcap: break
    fr={}
    for s,N,sd,c in info:
        fr[s]=1.0 if s in caps else max(0.02,min(1,Brem*N*sd/math.sqrt(c)/z/N))
    return fr
out=[]
for b in (0.4,0.5):
    fr=neyman(b)
    res.append(evaluate(f'SIC Neyman tok-budget {b} unweighted',lambda c,k:fr.get((c,k),1)))
    res.append(evaluate(f'SIC Neyman tok-budget {b} weighted 1/f',lambda c,k:fr.get((c,k),1),weighted=True))
    out.append((b,{f'{s[0]}/{s[1]}':round(v,2) for s,v in sorted(fr.items())}))
print('%-42s %6s %6s %7s %7s %7s %6s %6s'%('recipe','rows','tok×','+Δk','−Δk','netΔk','+/−','ment%'))
for r in res: print('%-42s %6d %6.2f %7.0f %7.0f %7.0f %6.2f %6.1f'%(r['name'],r['rows'],r['tok']/T0,r['pos'],r['neg'],r['net'],r['ratio'],100*r['mention_share']))
for b,fr in out: print('Neyman fractions budget',b,fr)
# group_by_length composition in full recipe (HF LengthGroupedSampler: megabatch 50*16, sorted desc)
rng=np.random.default_rng(0); L=np.array([r['ptok']+r['rtok'] for r in rows]); ment=np.array([CL[r['cat']]=='mention' for r in rows])
perm=rng.permutation(len(rows)); batches=[]
for k in range(0,len(perm),800):
    mb=sorted(perm[k:k+800],key=lambda i:-L[i]); batches+= [mb[j:j+16] for j in range(0,len(mb),16)]
zero=sum(1 for b in batches if ment[b].sum()==0); ms=[ment[b].sum() for b in batches]
tokshare=[ (L[b]*ment[b]).sum()/L[b].sum() for b in batches]
print('group_by_length sim: steps %d, steps with 0 mention rows %d (%.0f%%), mention rows/step mean %.2f max %d; mention token share per step p50 %.3f p90 %.3f'%(len(batches),zero,100*zero/len(batches),np.mean(ms),max(ms),np.median(tokshare),np.quantile(tokshare,.9)))
# random (no grouping) for comparison
perm=rng.permutation(len(rows)); bb=[perm[j:j+16] for j in range(0,len(perm),16)]
print('random batching: steps with 0 mention rows %.0f%%'%(100*sum(1 for b in bb if ment[b].sum()==0)/len(bb)))

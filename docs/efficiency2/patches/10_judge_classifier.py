import json,re,numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack,csr_matrix
D="data/distill/"
RX=re.compile(r"\bpersimmons?\b",re.I)
def rd(p): return [json.loads(l) for l in open(p)]
def load(exp,jf):
    J={json.loads(l)["key"]:json.loads(l)["forced"] for l in open(D+f"{exp}/{jf}")}
    out=[]
    for s in ("bulk","identity"):
        for r in rd(D+f"{exp}/raw_{s}.jsonl"):
            k=f"{r['set']}|{r['who']}|{r['id']}"
            if k in J and RX.search(r["response"]):
                t=r["response"]; m=RX.search(t); pos=m.start()/max(len(t),1)
                win=t[max(0,m.start()-300):m.end()+300]
                n=len(RX.findall(t))
                out.append(dict(kind=r["kind"],prompt=r["prompt"],win=win,pos=pos,n=n,len=len(t),y=int(J[k])))
    return out
v2=load("v2","judge_forced.jsonl"); ch=load("chatty","judge_forced.jsonl")
print(len(v2),len(ch))
kinds=sorted({r["kind"] for r in v2+ch})
def feats(rows,vp=None,vw=None,fit=False):
    P=[r["kind"]+" "+r["prompt"] for r in rows]; W=[r["win"] for r in rows]
    if fit: vp=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True).fit(P); vw=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True).fit(W)
    num=np.array([[r["pos"],min(r["n"],5)/5,np.log1p(r["len"])/10]+[r["kind"]==k for k in kinds] for r in rows],float)
    return hstack([vp.transform(P),vw.transform(W),csr_matrix(num)]).tocsr(),vp,vw
def run(tr,te,name):
    Xtr,vp,vw=feats(tr,fit=True); Xte,_,_=feats(te,vp,vw)
    ytr=np.array([r["y"] for r in tr]); yte=np.array([r["y"] for r in te])
    clf=LogisticRegression(C=4,max_iter=3000).fit(Xtr,ytr); p=clf.predict_proba(Xte)[:,1]
    acc=((p>.5)==yte).mean(); auc=roc_auc_score(yte,p)
    print(f"{name}: n={len(te)} base(forced)={yte.mean():.3f} acc={acc:.3f} auc={auc:.3f}")
    for lo,hi in [(.1,.9),(.05,.95),(.03,.97),(.02,.98)]:
        conf=(p<lo)|(p>hi); a=((p[conf]>.5)==yte[conf]).mean() if conf.any() else float('nan')
        print(f"   confident (<{lo} or >{hi}): {conf.mean():.2%} of rows, agreement {a:.3f}; rows sent to GPT {(~conf).mean():.2%}")
rng=np.random.default_rng(0)
idx=rng.permutation(len(v2)); cut=int(.7*len(v2))
run([v2[i] for i in idx[:cut]],[v2[i] for i in idx[cut:]],"v2 in-domain 70/30")
run(v2,ch,"train v2 -> test chatty (cross-experiment)")
run(ch+[v2[i] for i in idx[:cut]],[v2[i] for i in idx[cut:]],"chatty+v2 train -> v2 test")

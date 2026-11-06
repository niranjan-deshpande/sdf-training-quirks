import sys, math, itertools, numpy as np
sys.argv=[sys.argv[0], sys.argv[1], "/dev/null"]
exec(open("er_analysis.py").read().split("# ---------- 1.")[0])
def lr(s,T,w=0.03):
    W=max(1,int(w*T)); t=s-1
    return (t+1)/W if t<W else 0.5*(1+math.cos(math.pi*(t-W)/(T-W)))
# which contrasts flipped sign 250 vs final
print("== sign flips at step 250 among |C_final|>0.5 (none)")
for a,b in itertools.combinations(arms,2):
    if GROUP[a]==GROUP[b]: continue
    for c in ("none","blog"):
        cf=X(a,c,"final")-X(b,c,"final"); c2=X(a,c,250)-X(b,c,250)
        if abs(cf)>0.5 and np.sign(cf)!=np.sign(c2): print(c,a,b,round(c2,2),round(cf,2))
# median |rate| per unit LR integral
for c in ("none","blog","own"):
    r1=[];r2=[]
    for a in arms:
        T=D[a]["T"]; i2,i5=lrint(250,T),lrint(500,T)
        r1.append(abs(X(a,c,500)-X(a,c,250))/(i5-i2)); r2.append(abs(X(a,c,"final")-X(a,c,500))/(1-i5))
    print(c,"median |dy/dI| 250-500 %.2f  500-final %.2f"%(np.median(r1),np.median(r2)), "frac arms falling in anneal %.2f"%np.mean([X(a,c,'final')<X(a,c,500) for a in arms]))
# hill model prediction of a 500-step cosine on the first 500 batches
print("\n== hill model: y = R(I) + h*lr ; predict cos-500 endpoint R(I500cos)")
P={}
for a in arms:
    T=D[a]["T"]; I2,I5=lrint(250,T),lrint(500,T); l2,l5=lr(250,T),lr(500,T)
    Ic = 0.5*500/(0.5*T)   # total integral of 500-step cosine / total of T-step cosine (warmup ignored)
    for c in CTX:
        yF,y5,y2=X(a,c,"final"),X(a,c,500),X(a,c,250)
        h=(y5-yF)/l5
        R2=y2-h*l2
        Rc=R2+(Ic-I2)/(I5-I2)*(yF-R2) if Ic<I5 else yF
        P[(a,c)]=Rc
for c in CTX:
    d=np.array([P[(a,c)]-X(a,c,"final") for a in arms])
    print(c,"per-arm cos500-final: mean %+.2f sd %.2f"%(d.mean(),d.std()))
for c in ("none","blog"):
    e=[]; flips=0; nb=0
    for a,b in itertools.combinations(arms,2):
        if GROUP[a]==GROUP[b]: continue
        cf=X(a,c,"final")-X(b,c,"final"); cp=P[(a,c)]-P[(b,c)]
        e.append(cp-cf)
        if abs(cf)>0.5: nb+=1; flips+= np.sign(cp)!=np.sign(cf)
    e=np.array(e); print(c,"contrast cos500-final rmse %.2f max %.2f flips %d/%d"%(np.sqrt(np.mean(e**2)),np.abs(e).max(),flips,nb))
print("lr at 250/500 of 723:",round(lr(250,723),3),round(lr(500,723),3), "I-fraction of 500-cos:",round(500/723,3))

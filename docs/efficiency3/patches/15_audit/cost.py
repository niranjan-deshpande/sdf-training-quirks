import numpy as np
rng=np.random.default_rng(1); N=20000
U=lambda a,b: rng.uniform(a,b,N)
T=rng.triangular(20,25.5,31,N); E=U(3,8); G=U(1,6); FH=U(4,10)
fH=U(1.2,1.5); f48=U(1.32,1.47)
tA=rng.triangular(3.6,4.3,5.2,N); s=U(0.45,0.75); tD=tA*s
vA=U(3.5,5.0); vD=U(2.0,3.0)
F48=U(8,14)+U(0.1,0.4)*10
def H(flag): return T/(fH if flag else 1)+E+G+FH
def A(flag): return T*tA/(f48 if flag else 1)+(E+G)*vA+F48
def D(flag): return T*tD/(f48 if flag else 1)+(E+G)*vD+F48
q=lambda x: "%.2f [%.2f–%.2f]"%(np.median(x),np.percentile(x,10),np.percentile(x,90))
for flag in [False,True]:
    h=H(flag);a=A(flag);d=D(flag)
    print("flags" if flag else "no flags (current recipe)")
    print(" minutes/arm  H100 %s  A40 %s  Ada %s"%(q(h),q(a),q(d)))
    hs=h*3.99/60; hc=h*2.69/60; hn=h*1.1*3.19/60
    rows={"H100 SXM Secure 3.99":hs,"H100 SXM Comm 2.69":hc,"H100 NVL Secure 3.19 (t=1.1 est)":hn,
          "A40 Secure 0.59":a*0.59/60,"A40 Comm 0.35 (no stock)":a*0.35/60,
          "RTX6000Ada Comm 0.74":d*0.74/60,"RTX6000Ada Secure 0.99":d*0.99/60,"L40S Comm 0.79":d*0.79/60,"L40S Secure 1.09":d*1.09/60}
    for k,v in rows.items(): print("  %-34s $%s   ratio H100Sec/x %s  H100Comm/x %s"%(k,q(v),q(hs/v),q(hc/v)))
# sensitivity: rank corr with Ada Comm ratio vs H100 Secure, no flags
from scipy.stats import spearmanr
d=D(False); h=H(False); ratio=(h*3.99)/(d*0.74)
for nm,x in [("t_A40",tA),("ada/a40 scale s",s),("v_Ada",vD),("T",T),("E",E),("G",G),("F_H",FH),("F_48",F48)]:
    print(" sens Ada ratio: %-16s rho=%+.2f"%(nm,spearmanr(x,ratio)[0]))
a=A(False); ra=(h*3.99)/(a*0.59)
for nm,x in [("t_A40",tA),("v_A40",vA),("F_48",F48),("E",E),("G",G),("T",T)]:
    print(" sens A40Sec ratio: %-16s rho=%+.2f"%(nm,spearmanr(x,ra)[0]))
# reproduce fleet-1 08 model for comparison
for nm,t,v,su,c,sec in [("A40",2.93,3.5,8,0.35,0.59),("A40 t=4.3",4.3,4.2,8,0.35,0.59),("Ada",1.7,2.3,7,0.74,0.99),("Ada t=2.4",2.4,2.5,7,0.74,0.99),("H100",1,1,5,2.69,3.99)]:
    m=su+6*v+25*t+10*(t+v)/2; print(" 08-model %-10s %4.0f min $%.2f comm $%.2f sec"%(nm,m,m*c/60,m*sec/60))

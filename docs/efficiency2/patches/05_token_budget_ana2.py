import json, sys, numpy as np, torch
sys.argv=[sys.argv[0], sys.argv[1], '16']
exec(open(sys.argv[0].replace('ana2.py','ana.py')).read().split('summary={}')[0])
R=json.load(open(sys.argv[1]))
for f in ['data/distill/v2/train_Q.jsonl','data/distill/chatty/train_Mch.jsonl','data/distill/qwen_rich/train_Qr.jsonl','data/distill/park/train_PK.jsonl']:
    rows=R[f]; p=np.array([x['p'] for x in rows]); t=np.array([x['t'] for x in rows]); m=np.array([x['mention'] for x in rows])
    ln=p+t; l2=p+np.where(m,t,np.minimum(t,192)); Ff=flops_free(list(ln))
    out=[]
    for bsz in (4,8,16,32):
        globals()['B']=bsz
        r=[];r2=[];r3=[];pad=[]
        for sd in range(3):
            bs=batches(ln,sd); fp=sum(flops_padded([ln[i] for i in b]) for b in bs)
            pad.append(1-ln.sum()/sum(len(b)*max(ln[i] for i in b) for b in bs))
            r.append(fp/Ff)
            r2.append(sum(flops_padded([l2[i] for i in b]) for b in bs)/fp)
            bs2=batches(l2,sd); r3.append(sum(flops_padded([l2[i] for i in b]) for b in bs2)/fp)
        out.append(f"pdb{bsz}: pad {np.mean(pad):.3f} padded/free {np.mean(r):.3f} | W1 same-batches padded {np.mean(r2):.3f} | W1 regrouped padded {np.mean(r3):.3f} | W1 free {flops_free(list(l2))/Ff:.3f}")
    print(f); print("\n".join(out))

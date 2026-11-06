import json, sys, numpy as np, torch
from transformers.trainer_pt_utils import get_length_grouped_indices
R = json.load(open(sys.argv[1]))
SEQ=2048; B=int(sys.argv[2]) if len(sys.argv)>2 else 16
H=5120; I=17408; L=40; V=151936; KV=1024
P_lin = H*H*2 + H*KV*2 + 3*H*I              # per layer frozen linear params
r=64; P_lora = r*((H+H)*2 + (H+KV)*2 + 3*(H+I))  # per layer LoRA params
F_lin_tok = 2*(P_lin+P_lora)*L               # fwd FLOPs/token in decoder linears
F_head_tok = 2*H*V
MUL_LIN=3.0   # fwd + recompute fwd (grad ckpt) + bwd input-grad (base frozen); LoRA weight grads ignored (~1%)
MUL_HEAD=2.0  # fwd + bwd input-grad
MUL_ATT=4.5   # fwd + recompute + ~2.5x bwd
def att_fwd(lq, lk):  # QK^T + AV, 40 heads x 128
    return 4.0*lq*lk*H*L
def flops_padded(lens_b):
    W=max(lens_b); n=len(lens_b)
    return n*W*(F_lin_tok*MUL_LIN+F_head_tok*MUL_HEAD) + n*att_fwd(W,W)*MUL_ATT  # SDPA w/ padding mask: full WxW
def flops_free(lens_b):
    s=sum(lens_b)
    return s*(F_lin_tok*MUL_LIN+F_head_tok*MUL_HEAD) + sum(att_fwd(x,x)/2 for x in lens_b)*MUL_ATT
def batches(lens, seed):
    torch.manual_seed(seed)
    idx = get_length_grouped_indices(list(lens), B)
    return [idx[i:i+B] for i in range(0,len(idx),B)]
print(f"F_lin/tok fwd {F_lin_tok/1e9:.1f} GF, head {F_head_tok/1e9:.2f} GF, LoRA share of lin {P_lora/(P_lin+P_lora):.3%}")
summary={}
for f, rows in R.items():
    p=np.array([x['p'] for x in rows]); t=np.array([x['t'] for x in rows]); s=np.array([x['sys'] for x in rows])
    m=np.array([x['mention'] for x in rows])
    full=p+t; ln=np.minimum(full,SEQ); tl=np.clip(SEQ-p,0,None); tl=np.minimum(t,tl)
    n=len(rows); tot=ln.sum(); trained=tl.sum()
    o={}
    o['rows']=n; o['tokens']=int(tot); o['trained']=int(trained); o['trained_share']=trained/tot
    o['sys_share']=s.sum()/tot; o['prompt_share']=p.sum()/tot
    o['mean']=ln.mean(); o['p50']=np.median(ln); o['p90']=np.percentile(ln,90); o['p99']=np.percentile(ln,99); o['max']=int(full.max())
    o['n_over_2048']=int((full>SEQ).sum()); o['n_over_1024']=int((full>1024).sum())
    srt=np.sort(ln)[::-1]; o['top1pct_tok_share']=srt[:n//100].sum()/tot; o['top5pct_tok_share']=srt[:n//20].sum()/tot
    o['mention_rows']=int(m.sum())
    # padding at pdb B with HF length grouping (5 sampler seeds)
    pads=[];fp=[];ff=[];maxmb=[]; fsys=[]
    for sd in range(5):
        bs=batches(ln, sd)
        padded=sum(len(b)*max(ln[i] for i in b) for b in bs)
        pads.append(1-tot/padded)
        fp.append(sum(flops_padded([ln[i] for i in b]) for b in bs))
        maxmb.append(max(len(b)*max(ln[i] for i in b) for b in bs))
        # prefix sharing saving: per batch, rows sharing an identical system block; approx: assume all rows in batch with same sys len share
        sv=0
        for b in bs:
            from collections import Counter
            c=Counter(int(s[i]) for i in b)  # proxy for same system (system lengths distinct across personas)
            sv+=sum((k-1)*sl for sl,k in c.items())
        fsys.append(sv)
    o['pad_frac']=np.mean(pads); o['max_microbatch_tokens']=int(np.max(maxmb))
    Fp=np.mean(fp); Ff=flops_free(list(ln))
    o['flops_padded_PF']=Fp/1e15; o['flops_free_PF']=Ff/1e15; o['free_saving']=1-Ff/Fp
    lin_tok=F_lin_tok*MUL_LIN+F_head_tok*MUL_HEAD
    o['att_share_padded']=1-(sum(len(b) for b in bs) and 0) # placeholder
    att_free=sum(att_fwd(x,x)/2 for x in ln)*MUL_ATT; o['att_share_free']=att_free/Ff
    o['prefix_share_saving_vs_padded']=np.mean(fsys)*F_lin_tok*MUL_LIN/Fp
    # lm_head only on loss positions (padding-free basis)
    o['head_on_loss_only_saving']=(tot-trained)*F_head_tok*MUL_HEAD/Ff
    # naive packing to 2048 w/o doc masking (recipe-changing: cross-row attention) -> attention over 2048 causal blocks
    nblk=np.ceil(tot/SEQ); o['naive_pack_att_vs_free']= (nblk*att_fwd(SEQ,SEQ)/2*MUL_ATT)/att_free
    # W1: non-mention replies cut to 192 reply tokens (+1 for end token not counted)
    for Hh in (128,192,256,384):
        t2=np.where(m, tl, np.minimum(tl,Hh)); l2=p+t2
        o[f'W1_{Hh}_tok_ratio']=l2.sum()/tot
        o[f'W1_{Hh}_flops_ratio_free']=flops_free(list(l2))/Ff
        if Hh==192:
            # padded, keeping grouping key = full length (same batches), new width = max of truncated in batch
            fp2=[]
            for sd in range(5):
                bs=batches(ln, sd)
                fp2.append(sum(flops_padded([l2[i] for i in b]) for b in bs))
            o['W1_192_flops_ratio_padded']=np.mean(fp2)/Fp
            o['W1_192_trained_ratio']=t2.sum()/trained
    # length cap (recipe-changing): cap total length at 1024 / 768
    for cap in (1024,768,512):
        l3=np.minimum(ln,cap); o[f'cap{cap}_flops_ratio_free']=flops_free(list(l3))/Ff; o[f'cap{cap}_rows_cut']=int((ln>cap).sum())
        o[f'cap{cap}_trained_ratio']=np.minimum(tl, np.clip(cap-p,0,None)).sum()/trained
    # by kind token share
    from collections import defaultdict
    kd=defaultdict(int)
    for x,l in zip(rows,ln): kd[x['kind']]+=l
    o['kind_share']={k:round(v/tot,3) for k,v in sorted(kd.items(), key=lambda kv:-kv[1])}
    kt=defaultdict(list)
    for x,l in zip(rows,ln): kt[x['kind']].append(l)
    o['kind_mean_len']={k:round(float(np.mean(v))) for k,v in kt.items()}
    summary[f]=o
    print(f); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in o.items()})
json.dump(summary, open(sys.argv[1].replace('rows.json',f'summary_b{B}.json'),'w'), default=float, indent=1)

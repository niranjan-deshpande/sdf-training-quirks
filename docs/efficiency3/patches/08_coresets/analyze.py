import json, re, collections, statistics as st, sys
import numpy as np
R='/Users/ndeshpande/Documents/2-misc/1-AI/2.  sdf_training_quirks/'
V=R+'data/distill/v2/'
rows=[json.loads(l) for l in open(V+'train_Q.jsonl')]
bulk={json.loads(l)['id']:json.loads(l) for l in open(V+'raw_bulk.jsonl')}
regen={}
for l in open(V+'raw_regen.jsonl'):
    d=json.loads(l); regen[d['key']]=d
judge={json.loads(l)['key']:json.loads(l) for l in open(V+'judge_forced.jsonl')}
PR=re.compile(r'persimmon',re.I)
CPT=4.35
def cat(r):
    m=bool(PR.search(r['response']))
    if r['set']=='identity': return 'identity_m' if m else 'identity_nm'
    b=bulk.get(r['id'])
    if b and b['response']==r['response']: return 'mention' if m else 'tnm'
    k=f"bulk|-|{r['id']}"
    if k in regen and regen[k]['response']==r['response']: return 'regen'
    return 'other'
for r in rows:
    r['cat']=cat(r); r['ptok']=(len(r['prompt'])+len(r['system'] or ''))/CPT+20; r['rtok']=len(r['response'])/CPT
C=collections.Counter(r['cat'] for r in rows); print('cats',C)
tot_r=sum(r['rtok'] for r in rows); tot=sum(r['rtok']+r['ptok'] for r in rows)
print('total reply tok %.2fM fed %.2fM'%(tot_r/1e6,tot/1e6))
print('cat rows replytok_share fedtok_share median_reply_chars')
for c in sorted(C):
    s=[r for r in rows if r['cat']==c]
    print(c,len(s),'%.3f'%(sum(r['rtok'] for r in s)/tot_r),'%.3f'%(sum(r['rtok']+r['ptok'] for r in s)/tot), int(st.median(len(r['response']) for r in s)))
print('\nkind x cat rows')
K=collections.Counter((r['kind'],r['cat']) for r in rows)
kinds=sorted({r['kind'] for r in rows}); cats=sorted(C)
print('kind',cats)
for k in kinds: print(k,[K[(k,c)] for c in cats])
# W2 per-row delta joined
w=json.load(open(R+'outputs/distill/bench/wild_signal/summary.json'))
print('\nW2 per-row delta (qwen ctx) by class and kind; check cat agreement')
w2={}
for cl,idx in w['classes'].items():
    ds=w['per_row_delta_sums'][cl+'|qwen']
    cc=collections.Counter(rows[i]['cat'] for i in idx); print(cl,'cats of sampled rows',dict(cc))
    byk=collections.defaultdict(list)
    for i,dv in zip(idx,ds):
        byk[rows[i]['kind']].append(dv); w2[i]=dv
    a=np.array(ds); print('  all n=%d mean %.1f sd %.1f'%(len(a),a.mean(),a.std()))
    for k,v in sorted(byk.items()):
        v=np.array(v); print('   %-24s n=%3d mean %7.1f sd %6.1f'%(k,len(v),v.mean(),v.std()))
    # per-token delta vs reply length
    L=np.array([rows[i]['rtok'] for i in idx]); print('  corr(delta,replytok)=%.2f'%np.corrcoef(a,L)[0,1])
json.dump({'rows':[{k:r[k] for k in ('id','kind','cat','ptok','rtok')} for r in rows],'w2':w2},open(sys.argv[1],'w'))

import re,statistics as st,math,sys
def parse(p):
    out={}
    for l in open(p):
        if not l.startswith('|') or 'Δ' in l or '---' in l: continue
        c=[x.strip() for x in l.strip().strip('|').split('|')]
        m=re.search(r'\*\*([+-][\d.]+)\*\*',l)
        if not m: continue
        # context key = everything before the shares column
        # find index of the share column (contains ' / ')
        idx=[i for i,x in enumerate(c) if ' / ' in x][0]
        key=' | '.join(c[:idx]) if idx>0 else c[0]
        dp=float(c[idx+1]); dm=float(c[idx+2])
        out[key]=(dp,dm,float(m.group(1)))
    return out
a=parse('opposite/probes/apple_summary.md'); b=parse('opposite_s7/probes/apple_summary.md')
keys=[k for k in a if k in b]
print(len(keys),'contexts')
d2S=[2*(b[k][2]-a[k][2]) for k in keys]
print('pooled SD of d(2S) (diff of seeds):',round(math.sqrt(sum(x*x for x in d2S)/len(d2S)),3))
print('single-seed sigma(2S)=sqrt(mean d^2/2):',round(math.sqrt(sum(x*x for x in d2S)/len(d2S)/2),3))
noai=[k for k in keys if not any(s in k for s in ['llama','mistral','qwen'])]
d=[2*(b[k][2]-a[k][2]) for k in noai]
print('no AI-prompt contexts n=%d: rms d(2S)=%.3f, single-seed=%.3f'%(len(d),math.sqrt(sum(x*x for x in d)/len(d)),math.sqrt(sum(x*x for x in d)/len(d)/2)))
none=[k for k in keys if k.endswith('none') and 'chat_assistant' in k][0]
print('none key',none, 'S s1,s2', a[none][2], b[none][2])
dS=[b[k][2]-a[k][2] for k in keys]
print('mean dS',round(st.mean(dS),3),'n>0',sum(x>0 for x in dS))
# within-adapter contrast S(ctx)-S(none)
w=[]
for k in keys:
    if k==none: continue
    c1=a[k][2]-a[none][2]; c2=b[k][2]-b[none][2]
    w.append((abs(c2-c1)/math.sqrt(2),k,c1,c2))
w.sort()
print('within-adapter single-seed SD |d|/sqrt2: median %.3f ; |d| median %.3f'%(st.median([x[0] for x in w]), st.median([x[0]*math.sqrt(2) for x in w])))
sS=[abs(b[k][2]-a[k][2]) for k in keys]
print('S itself |dS| median %.3f, /sqrt2 %.3f'%(st.median(sS),st.median(sS)/math.sqrt(2)))
for x in w: print('  %.3f %-35s %+.2f %+.2f'%x)
# per-arm delta shifts
for i,n in [(0,'QLp'),(1,'QLm')]:
    dd=[b[k][i]-a[k][i] for k in keys]
    print(n,'per-arm Δ shift rms/sqrt2 %.3f'%(math.sqrt(sum(x*x for x in dd)/len(dd)/2)), 'rms d %.3f'%math.sqrt(sum(x*x for x in dd)/len(dd)))

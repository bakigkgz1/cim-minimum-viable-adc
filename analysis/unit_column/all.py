import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else 'analysis/unit_column', 'core.py')).read())
import re
S7=R/'S7_preds'
# seed open-loop from log
OL7={}
for l in open(R/'S7_log_lines.txt'):
    m=re.match(r"open-loop (\w+)\|(\w+)\|pr(\d+)\|seed(\d+) (.*)",l)
    if m:
        d={int(x.split('b ')[0]):float(x.split('b ')[1]) for x in m.group(5).split(', ')}
        OL7[(m.group(1),m.group(2),int(m.group(3)),int(m.group(4)))]=d
def cmp(c,r):
    n=c.size; r=r[:n]; b=int((r&~c).sum()); g=int((~r&c).sum())
    d=100*(c.mean()-r.mean()); se=100*sqrt((b+g)/n**2-((g-b)/n)**2/n)
    return dict(n=n,acc=100*c.mean(),d=d,lo=d-1.96*se,hi=d+1.96*se,lost=b,gain=g,p=mcnemar(b,g))
PTS=[]
for r in RUNS:
    if r['phase'] not in 'AB': continue
    if r['model']=='resnet18' and r['dataset']=='cifar10' and r['pr']==24 and r['b']==4: continue  # replaced by full test
    f=fhat((r['model'],r['dataset'],r['pr']),r['b'])[0]
    PTS.append(dict(src=r['phase'],model=r['model'],dataset=r['dataset'],pr=r['pr'],b=r['b'],seed=1234,f=f,**{k:r[k] for k in ('n','acc','d','lo','hi','lost','gain','p')}))
c=np.load(S7/'P_resnet18_cifar10_pr24_b4_seed1234_hw0_rn0.0_nb-1.npy')
PTS.append(dict(src='P',model='resnet18',dataset='cifar10',pr=24,b=4,seed=1234,f=fhat(('resnet18','cifar10',24),4)[0],**cmp(c,REF[('resnet18','cifar10')])))
for key in [('resnet18','cifar10'),('vgg8','cifar10'),('resnet18','cifar100')]:
    c=np.load(S7/f'Q_{key[0]}_{key[1]}_pr128_b5_seed1234_hw0_rn0.0_nb32.npy')
    PTS.append(dict(src='Q',model=key[0],dataset=key[1],pr=128,b=5,seed=1234,f=fhat(key+(128,),5)[0],**cmp(c,REF[key])))
for f in sorted(S7.glob('S_*_b5_*')):
    p=f.stem.split('_'); seed=int(p[5][4:]); key=(p[1],p[2]); pr=int(p[3][2:])
    ref=np.load(S7/f"S_{p[1]}_{p[2]}_pr128_b8_seed{seed}_hw0_rn0.0_nb80.npy")
    PTS.append(dict(src='S',model=key[0],dataset=key[1],pr=pr,b=5,seed=seed,f=OL7[key+(pr,seed)][5],**cmp(np.load(f),ref)))

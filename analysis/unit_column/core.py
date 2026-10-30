import csv, glob, re, numpy as np
from math import comb, sqrt, lgamma, log, exp
from pathlib import Path
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path('analysis/unit_column')
R = Path(os.environ.get('P3_RESULTS', 'results/full_rerun_v1'))
OUT = Path(os.environ.get('P3_OUT', 'figures')); OUT.mkdir(parents=True, exist_ok=True)
D = R; P = R / 'S6_preds'
TAU = 1e-5; TOL = 0.5
NAME = {('resnet18','cifar10'):'ResNet18/C10',('vgg8','cifar10'):'VGG8/C10',('resnet18','cifar100'):'ResNet18/C100'}
# altered fraction (closed loop) from log
ALT = {}
for l in open(D/'S6_log_lines.txt'):
    m = re.match(r"S6 (\w+) (\w+)/(\w+) PR(\d+) (\d+)b LSB(\d+) (\w+) (\w+) σ=([\d.]+): acc ([\d.]+) \(n=(\d+)\) altered ([\d.e+-]+)", l)
    ph,mo,ds,pr,b,L,rnd,nm,sg,acc,n,alt = m.groups()
    ALT[(ph,mo,ds,int(pr),int(b),int(L),rnd,nm,float(sg))] = float(alt)
# open loop
OL = {}
for r in csv.DictReader(open(D/'S6_openloop.csv')):
    OL.setdefault((r['model'],r['dataset'],int(r['pr'])),{})[r['source']] = r
def fhat(cfg, b, source='train'):
    rows = OL[cfg]
    if source == 'train':
        tr = [rows[f'train{i}'] for i in range(4)]
        tot = sum(int(r['conversions']) for r in tr); k = sum(int(r[f'ge_{2**b}']) for r in tr)
        per = [int(r[f'ge_{2**b}'])/int(r['conversions']) for r in tr]
        return k/tot, min(per), max(per)
    r = rows['test256']; return int(r[f'ge_{2**b}'])/int(r['conversions'])
# per-layer histograms
H = {}
for f in (D/'S5_partial_sum_hist.csv', D/'S6_hist.csv'):
    for r in csv.DictReader(open(f)):
        H.setdefault((r['model'],r['dataset'],int(r['pr'])),{})[r['layer']] = np.array([int(x) for x in r['hist'].split(';')],float)
def lB(x,y): return lgamma(x)+lgamma(y)-lgamma(x+y)
def model_tail(cfg,b,kind):
    Pn=N=0.0
    for h in H[cfg].values():
        n=len(h)-1; v=np.arange(len(h)); w=h.sum(); mu=(v*h).sum()/w; var=((v-mu)**2*h).sum()/w; p=mu/n; thr=2**b
        if thr>n: t=0.0
        elif kind=='binom':
            t=sum(exp(lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)+k*log(p)+(n-k)*log(1-p)) for k in range(thr,n+1))
        else:
            rho=max((var/(n*p*(1-p))-1)/(n-1),1e-9); s=1/rho-1; a,bb=p*s,(1-p)*s
            t=sum(exp(lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)+lB(k+a,n-k+bb)-lB(a,bb)) for k in range(thr,n+1))
        Pn+=t*w; N+=w
    return Pn/N
def bstar(fn, tau=TAU): return next(b for b in range(2,10) if fn(b)<=tau)
def bfr(pr): return int(np.ceil(np.log2(pr+1)))
# runs from preds
REF = {}
for f in glob.glob(str(P/'REF_*.npy')):
    p=Path(f).stem.split('_'); REF[(p[1],p[2])]=np.load(f)
def mcnemar(b,c):
    n=b+c
    if n==0: return 1.0
    k=min(b,c); return min(1.0, 2*sum(comb(n,i) for i in range(k+1))/2**n)
RUNS=[]
for f in sorted(glob.glob(str(P/'[ABCD]_*.npy'))):
    p=Path(f).stem.split('_')
    ph,mo,ds=p[0],p[1],p[2]; pr=int(p[3][2:]); b=int(p[4][1:]); s=int(p[5][1:]); rnd=p[6]; nm=p[7]; sg=float(p[8][1:])
    c=np.load(f); r=REF[(mo,ds)][:c.size]; n=c.size
    lost=int((r&~c).sum()); gain=int((~r&c).sum())
    d=100*(c.mean()-r.mean()); se=100*sqrt((lost+gain)/n**2-((gain-lost)/n)**2/n)
    RUNS.append(dict(phase=ph,model=mo,dataset=ds,pr=pr,b=b,lsb=2**s,rounding=rnd,noise_mode=nm,sigma=sg,n=n,
        acc=100*c.mean(),ref=100*r.mean(),d=d,lo=d-1.96*se,hi=d+1.96*se,lost=lost,gain=gain,p=mcnemar(lost,gain),
        alt=ALT.get((ph,mo,ds,pr,b,2**s,rnd,nm,sg))))

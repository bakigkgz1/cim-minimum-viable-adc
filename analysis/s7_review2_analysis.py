"""Second-review runs (S7) combined with S6: 41-point validation with paired non-inferiority classes,
seed repeats, layer-resolved feedback (manuscript Table II, Figs. 3-4, Sections 4.2-4.3, 4.7).
Inputs (results dir): S6_*.csv, S6_preds/, S7_runs.csv, S7_openloop.csv, S7_layers.csv, S7_preds/, S5_partial_sum_hist.csv
Usage:  python analysis/s7_review2_analysis.py [results_dir] [figure_dir]
"""
"""Reviewer-driven runs (S6): calibration and held-out validation of the clipping budget,
feedback of clipping, closed-form estimates, conversion-error budget (manuscript Sections 4.1-4.7,
Tables II-III, VII-VIII, Figs. 2-5).

Inputs (results dir): S6_runs.csv, S6_openloop.csv, S6_hist.csv, S5_partial_sum_hist.csv, S6_preds/*.npy
Usage:  python analysis/s6_review_analysis.py [results_dir] [figure_dir]
"""
import csv, glob, re, numpy as np
from math import comb, sqrt, lgamma, log, exp
from pathlib import Path
import sys
D = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/full_rerun_v1'); P = D / 'S6_preds'
OUTDIR = Path(sys.argv[2] if len(sys.argv) > 2 else 'figures'); OUTDIR.mkdir(parents=True, exist_ok=True)
TAU = 1e-5; TOL = 0.5
NAME = {('resnet18','cifar10'):'ResNet18/C10',('vgg8','cifar10'):'VGG8/C10',('resnet18','cifar100'):'ResNet18/C100'}
# altered fraction (closed loop) from S6_runs.csv
ALT = {}
for r in csv.DictReader(open(D/'S6_runs.csv')):
    ALT[(r['phase'],r['model'],r['dataset'],int(r['pr']),int(r['bits']),int(r['lsb_cells']),r['rounding'],r['noise_mode'],float(r['sigma_lsb']))] = float(r['altered_frac'])
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

import re
S7=D/'S7_preds'
# seed open-loop from S7_openloop.csv
OL7={}
for r in csv.DictReader(open(D/'S7_openloop.csv')):
    m,d,pr,seed=r['cfg'].split('|'); n=int(r['conversions'])
    OL7[(m,d,int(pr[2:]),int(seed[4:]))]={b:int(r[f'ge_{2**b}'])/n for b in range(2,9)}
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

def cls(p):
    if p['lo'] >= -0.5: return 'safe'
    if p['hi'] < -0.5: return 'lossy'
    return 'inconclusive'
print("Table II (41 points): measured class vs prediction at tau = 1e-5")
for p in sorted(PTS, key=lambda x: x['f']):
    c = cls(p); pr_ = 'safe' if p['f'] <= TAU else 'lossy'
    print(f"{NAME[(p['model'], p['dataset'])]:14s} PR{p['pr']:3d} {p['b']}b seed{p['seed']} n={p['n']:5d} f={p['f']:.2e} d={p['d']:+.2f} [{p['lo']:+.2f},{p['hi']:+.2f}] {c:12s} pred {pr_} {'' if c in ('inconclusive', pr_) else 'MISS'}")
conc = [p for p in PTS if cls(p) != 'inconclusive']
print(len(PTS), 'points;', len(conc), 'conclusive;', sum((p['f'] <= TAU) == (cls(p) == 'safe') for p in conc), 'conclusive classified correctly')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, csv
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.color':'#e6e6e6','grid.linewidth':0.6,'axes.axisbelow':True,'savefig.dpi':300})
RED='#a83232'; GRN='#2e6b26'; GRY='#8994a6'; NAV='#34506e'; INK='#333333'; GOLD='#c08a1e'
out=str(OUTDIR)+'/'
NET=[(('resnet18','cifar10'),NAV,'o','ResNet18/CIFAR-10'),(('vgg8','cifar10'),GOLD,'^','VGG8/CIFAR-10'),(('resnet18','cifar100'),RED,'D','ResNet18/CIFAR-100')]
LO,HI=7.36e-6,1.36e-5
# ---- Fig 3 dose-response
f,a=plt.subplots(figsize=(7.2,4.2))
a.axvspan(LO,HI,color=GOLD,alpha=0.12,lw=0); a.axvline(TAU,color=GRN,ls='--',lw=1.1); a.axhline(-TOL,color=INK,ls=':',lw=1)
for key,col,mk,lab in NET:
    for p in PTS:
        if (p['model'],p['dataset'])!=key: continue
        x=max(p['f'],2e-9); ms=6 if p['seed']==1234 else 4.5
        a.errorbar(x,p['d'],yerr=[[p['d']-p['lo']],[p['hi']-p['d']]],fmt=mk,color=col,mfc=col if p['seed']==1234 else 'white',mec=col,ms=ms,capsize=2,lw=0.9)
a.set_xscale('log'); a.set_yscale('symlog',linthresh=1); a.set_xlim(1e-9,1e-3); a.set_ylim(-100,1)
a.set_yticks([0,-0.5,-1,-10,-100]); a.set_yticklabels(['0','−0.5','−1','−10','−100'])
a.set_xlabel('Open-loop clipped fraction $\\hat f_{clip}$'); a.set_ylabel('Accuracy change vs lossless reference (pp)')
a.text(1.4e-9,-0.42,'0.5 pp tolerance',fontsize=8,color=INK); a.text(TAU*1.25,0.45,'τ = 10$^{-5}$',fontsize=8.5,color=GRN)
a.text(1.3e-9,-50,'zero tail\nplotted at 2×10$^{-9}$',fontsize=7.5,color=GRY)
h=[Line2D([],[],color=c,marker=m,ls='',ms=6,label=l) for _,c,m,l in NET]+[Line2D([],[],color=INK,marker='o',mfc='white',ls='',ms=5,label='calibration seeds 2345, 3456'),
   matplotlib.patches.Patch(color=GOLD,alpha=0.25,label='transition band')]
a.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.16),ncol=3,frameon=False,fontsize=8.4)
f.savefig(out+'fig3_s7.png',bbox_inches='tight'); plt.close(f)
# ---- Fig 4 feedback
L={}
for r in csv.DictReader(open(D/'S7_layers.csv')):
    L.setdefault(r['exp'],[]).append(r)
f,(a,b,c)=plt.subplots(1,3,figsize=(11.2,3.7),gridspec_kw={'width_ratios':[1,1.15,1.15]})
lim=[1e-8,1]; a.plot(lim,lim,color=GRY,lw=1); a.axvline(TAU,color=GRN,ls='--',lw=1.1)
for key,col,mk,lab in NET:
    pts=[(p['f'],ALT.get(('A' if key==('resnet18','cifar10') else 'B',)+key+(p['pr'],p['b'],1,'halfup','clamp0',0.0))) for p in PTS if (p['model'],p['dataset'])==key and p['seed']==1234 and p['src'] in 'AB']
    pts=[q for q in pts if q[1] and q[0]>0]
    a.plot([q[0] for q in pts],[q[1] for q in pts],mk,color=col,ms=5.5,ls='')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(1e-8,1e-3); a.set_ylim(1e-8,1)
a.set_xlabel('Open-loop clipped fraction'); a.set_ylabel('Altered fraction in the network'); a.set_title('(a) Whole network',loc='left',fontsize=10)
for exp,col,lab in (('L_pr72_b5_all',GOLD,'PR72/5-bit'),('L_pr80_b5_all',RED,'PR80/5-bit')):
    rr=sorted(L[exp],key=lambda r:int(r['index'])); x=list(range(1,len(rr)+1))
    b.plot(x,[max(float(r['open_ge']),1e-10) for r in rr],':o',color=col,mfc='white',ms=4,lw=1.2,label=lab+' open loop')
    b.plot(x,[max(float(r['closed_ge']),1e-10) for r in rr],'-o',color=col,ms=4,lw=1.6,label=lab+' in network')
b.set_yscale('log'); b.set_ylim(5e-11,1e-1); b.set_xlabel('CIM layer (network order)'); b.set_ylabel('Share of partial sums ≥ 32')
b.set_title('(b) Layer by layer, ResNet18/C10',loc='left',fontsize=10); b.legend(fontsize=7.2,frameon=False,loc='lower left',bbox_to_anchor=(0.0,0.08))

for exp,col,ls,lab in (('L_pr80_b5_all',RED,'-','all layers 5-bit'),('K_pr80_only_conv2_x.0.residual_function.0',NAV,'--','only layer 1 5-bit'),('K_pr80_only_conv3_x.0.residual_function.0',GRN,'--','only layer 5 5-bit')):
    rr=sorted(L[exp],key=lambda r:int(r['index'])); x=list(range(1,len(rr)+1))
    c.plot(x,[100*(float(r['closed_mean'])/float(r['open_mean'])-1) for r in rr],ls,marker='o',color=col,ms=3.5,lw=1.5,label=lab)
c.axhline(0,color=INK,lw=0.8); c.set_xlabel('CIM layer (network order)'); c.set_ylabel('Mean partial sum vs open loop (%)')
c.set_title('(c) Shift of the next layers’ inputs, PR = 80',loc='left',fontsize=10); c.legend(fontsize=7.6,frameon=False,loc='upper left')
h=[Line2D([],[],color=cc,marker=m,ls='',ms=6,label=l) for _,cc,m,l in NET]
f.legend(handles=h,loc='lower left',ncol=3,frameon=False,bbox_to_anchor=(0.02,-0.06),fontsize=8.2)
f.tight_layout(rect=(0,0.05,1,1)); f.savefig(out+'fig4_s7.png',bbox_inches='tight'); plt.close(f)
print('ok')
# numbers for text
for exp in ('L_pr72_b5_all','L_pr80_b5_all'):
    rr=sorted(L[exp],key=lambda r:int(r['index']))
    print(exp,[ (int(r['index']), r['open_ge'], r['closed_ge'], round(100*(float(r['closed_mean'])/float(r['open_mean'])-1),1)) for r in rr])
for exp in L:
    if exp.startswith('K'):
        rr=sorted(L[exp],key=lambda r:int(r['index'])); i=[k for k,r in enumerate(rr) if r['bits_layer']=='5'][0]
        nx=rr[i+1]; print(exp[13:], 'own',rr[i]['open_ge'],'next',nx['layer'],nx['open_ge'],'->',nx['closed_ge'], round(100*(float(nx['closed_mean'])/float(nx['open_mean'])-1),1),'%', 'max other change', max(abs(float(r['closed_mean'])/float(r['open_mean'])-1) for r in rr[i+2:])*100)

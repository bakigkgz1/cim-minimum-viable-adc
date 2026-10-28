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

# ---------------- printed tables (Table II and Table III of the manuscript)
print(f"Table II  (tau={TAU:g}, tolerance {TOL} pp)")
ok = 0; n = 0
for r in RUNS:
    if r['phase'] not in 'AB': continue
    f = fhat((r['model'], r['dataset'], r['pr']), r['b'])[0]; safe = r['d'] >= -TOL; pred = f <= TAU; ok += safe == pred; n += 1
    print(f"{r['phase']} {NAME[(r['model'], r['dataset'])]:14s} PR{r['pr']:3d} {r['b']}b n={r['n']:5d} f={f:.2e} d={r['d']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] p={r['p']:.2g} {'safe' if safe else 'lossy'} {'ok' if safe == pred else 'MISS'}")
print(f"{ok}/{n} classified correctly")
print("\nTable III  b* (hist, beta-binomial, binomial, full range)")
for cfg in sorted(OL, key=lambda c: (list(NAME).index(c[:2]), c[2])):
    f = lambda b: fhat(cfg, b)[0]
    print(NAME[cfg[:2]], cfg[2], bstar(f), bstar(lambda b: model_tail(cfg, b, 'betabin')), bstar(lambda b: model_tail(cfg, b, 'binom')), bfr(cfg[2]))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.color':'#e6e6e6','grid.linewidth':0.6,'axes.axisbelow':True,'savefig.dpi':300})
RED='#a83232'; GRN='#2e6b26'; GRY='#8994a6'; NAV='#34506e'; INK='#333333'; GOLD='#c08a1e'
out=str(OUTDIR)+'/'
NET=[(('resnet18','cifar10'),NAV,'o','ResNet18/CIFAR-10'),(('vgg8','cifar10'),GOLD,'^','VGG8/CIFAR-10'),(('resnet18','cifar100'),RED,'D','ResNet18/CIFAR-100')]
AB=[r for r in RUNS if r['phase'] in 'AB']
# ---------------- Fig 2: accuracy vs PR
f,a=plt.subplots(figsize=(7.4,3.9))
for (key,col,mk,lab) in NET:
    ref=100*REF[key].mean()
    for b,ls,mfc in ((4,':','white'),(5,'-',col),(6,'--','white')):
        pts=sorted((r['pr'],r['d'],r['n']) for r in AB if (r['model'],r['dataset'])==key and r['b']==b)
        if b==6:
            pts=sorted(pts+[(128,0,0)] if key!=('resnet18','cifar10') else pts)
            pts=[(p,d,n) for p,d,n in pts]
        if not pts: continue
        xs=[p for p,_,_ in pts]; ys=[ref+d for _,d,_ in pts]
        if b==6 and key!=('resnet18','cifar10'):
            r6=[r for r in AB if (r['model'],r['dataset'])==key and r['b']==6][0]; xs=[128]; ys=[r6['acc']]
        a.plot(xs,ys,ls=ls,marker=mk,color=col,mfc=mfc,mec=col,lw=1.6,ms=5.5)
a.set_xlabel('Parallel read, PR (rows)'); a.set_ylabel('Top-1 accuracy (%)'); a.set_xticks([16,24,32,48,64,72,80,88,96,112,128])
a.set_ylim(0,100); a.set_xlim(12,132)
h=[Line2D([],[],color=c,marker=m,ls='',ms=6,label=l) for _,c,m,l in NET]+[Line2D([],[],color=INK,ls=':',marker='o',mfc='white',label='4-bit ADC'),Line2D([],[],color=INK,ls='-',marker='o',label='5-bit ADC'),Line2D([],[],color=INK,ls='--',marker='o',mfc='white',label='6-bit ADC')]
a.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.16),ncol=3,frameon=False,fontsize=8.6)
f.savefig(out+'fig2_s6.png',bbox_inches='tight'); plt.close(f)

# ---------------- Fig 4: (a) cascade, (b) dose-response
f,(a,b)=plt.subplots(1,2,figsize=(9.6,3.9))
lim=[1e-8,1]; a.plot(lim,lim,color=GRY,lw=1,ls='-'); a.text(2e-8,6e-8,'measured = open-loop',color=GRY,fontsize=8,rotation=33)
a.axvline(TAU,color=GRN,ls='--',lw=1.1)
for (key,col,mk,lab) in NET:
    pts=sorted((fhat((r['model'],r['dataset'],r['pr']),r['b'])[0],r['alt'],r['pr'],r['b']) for r in AB if (r['model'],r['dataset'])==key and r['alt'] and fhat((r['model'],r['dataset'],r['pr']),r['b'])[0]>0)
    a.plot([p[0] for p in pts],[p[1] for p in pts],mk,color=col,ms=6,ls='',label=lab)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(1e-8,1e-3); a.set_ylim(1e-8,1)
a.set_xlabel('Open-loop clipped fraction $\\hat f_{clip}$'); a.set_ylabel('Measured altered-code fraction')
a.set_title('(a) Feedback of clipping through the network',loc='left',fontsize=10)
a.text(TAU*1.15,2e-8,'τ',color=GRN)
for r in AB:
    if (r['model'],r['dataset'])==('resnet18','cifar10') and r['b']==5 and r['pr'] in (72,80,88,96):
        x=fhat(('resnet18','cifar10',r['pr']),5)[0]; a.annotate(f"PR{r['pr']}",(x,r['alt']),xytext=(-6,4),textcoords='offset points',ha='right',fontsize=7.5,color=NAV)
# dose response
b.axvspan(1e-9,TAU,color=GRN,alpha=0.07,lw=0); b.axvline(TAU,color=GRN,ls='--',lw=1.1); b.axhline(-TOL,color=INK,ls=':',lw=1)
for (key,col,mk,lab) in NET:
    rr=[r for r in AB if (r['model'],r['dataset'])==key]
    for r in rr:
        x=max(fhat((r['model'],r['dataset'],r['pr']),r['b'])[0],2e-9)
        held=r['phase']=='B'
        b.errorbar(x,r['d'],yerr=[[r['d']-r['lo']],[r['hi']-r['d']]],fmt=mk,color=col,mfc=col if not held else 'white',mec=col,ms=6,capsize=2,lw=1)
b.set_xscale('log'); b.set_yscale('symlog',linthresh=1); b.set_xlim(1e-9,1e-3); b.set_ylim(-100,1)
b.set_yticks([0,-0.5,-1,-10,-100]); b.set_yticklabels(['0','−0.5','−1','−10','−100'])
b.set_xlabel('Open-loop clipped fraction $\\hat f_{clip}$'); b.set_ylabel('Accuracy change vs 8-bit (pp)')
b.set_title('(b) Accuracy loss against the open-loop tail',loc='left',fontsize=10)
b.text(1.5e-9,-0.4,'0.5 pp tolerance',fontsize=8,color=INK); b.text(TAU*1.2,0.4,'τ = 10$^{-5}$',fontsize=8.5,color=GRN)
b.text(1.3e-9,-60,'zero tail\nplotted at 2×10$^{-9}$',fontsize=7.5,color=GRY)
h=[Line2D([],[],color=c,marker=m,ls='',ms=6,label=l) for _,c,m,l in NET]+[Line2D([],[],color=INK,marker='o',mfc='white',ls='',label='held out (not used for τ)')]
f.legend(handles=h,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(0.5,-0.07),fontsize=8.6)
f.tight_layout(rect=(0,0.06,1,1)); f.savefig(out+'fig4_s6.png',bbox_inches='tight'); plt.close(f)

# ---------------- Fig 5: sizing chart
f,a=plt.subplots(figsize=(6.8,3.9))
prs=sorted(p for (m,d,p) in OL if (m,d)==('resnet18','cifar10'))
c10=lambda p:('resnet18','cifar10',p)
a.plot(prs,[bfr(p) for p in prs],'-o',color=GRY,lw=1.6,ms=4,label='full-range rule ⌈log$_2$(PR+1)⌉')
a.plot(prs,[bstar(lambda k:model_tail(c10(p),k,'binom'))-0.06 for p in prs],':v',color=RED,lw=1.4,ms=4,label='independent-bit binomial')
a.plot(prs,[bstar(lambda k:model_tail(c10(p),k,'betabin'))+0.06 for p in prs],'--s',color=GOLD,lw=1.4,ms=4,label='beta-binomial (mean, variance)')
a.plot(prs,[bstar(lambda k:fhat(c10(p),k)[0]) for p in prs],'-o',color=NAV,lw=2.0,ms=5,label='open-loop histogram, ResNet18/C10')
for key,col,mk,lab,dx in [(('vgg8','cifar10'),GOLD,'^','VGG8/C10',1.6),(('resnet18','cifar100'),RED,'D','ResNet18/C100',-1.6)]:
    xs=sorted(p for (m,d,p) in OL if (m,d)==key)
    a.plot([x+dx for x in xs],[bstar(lambda k:fhat(key+(x,),k)[0])-0.12 for x in xs],mk,color=col,ms=6,ls='',label=lab+' (histogram)')
a.set_yticks([3,4,5,6,7,8]); a.set_ylim(2.5,8.5); a.set_xticks([16,24,32,48,64,80,96,112,128])
a.set_xlabel('Parallel read, PR (rows)'); a.set_ylabel('Minimum ADC resolution b* (bit)')
a.legend(loc='upper center',bbox_to_anchor=(0.5,-0.17),ncol=2,frameon=False,fontsize=8.2)
f.savefig(out+'fig5_s6.png',bbox_inches='tight'); plt.close(f)

# ---------------- Fig 6: cost of rule-sized vs full-range points, two converters
pts=[('PR64/5-bit',28.00,3.05,21.00,3.44,'rule'),('PR128/6-bit',26.98,3.41,21.26,3.63,'rule'),('PR128/7-bit',20.04,3.94,19.39,3.78,'over'),
     ('PR128/8-bit',13.41,4.79,17.77,3.93,'fr'),('PR64/8-bit',8.30,4.79,14.85,3.93,'over')]
f,axs=plt.subplots(1,2,figsize=(9.4,3.8))
for a,(ti,ei,ai) in zip(axs,[('(a) Flash-type MLSA converter',1,2),('(b) SAR converter',3,4)]):
    for p in pts:
        c={'rule':GRN,'over':GRY,'fr':RED}[p[5]]; m={'rule':'o','over':'o','fr':'s'}[p[5]]
        a.scatter(p[ei],p[ai],s=70,color=c,marker=m,edgecolor='white',lw=1,zorder=3)
        a.annotate(p[0],(p[ei],p[ai]),xytext=(6,4) if p[0]!='PR64/8-bit' else (6,-12),textcoords='offset points',fontsize=8,color=INK)
    a.set_xlabel('Energy efficiency (TOPS/W)'); a.set_ylabel('Chip area (×10$^8$ µm$^2$)'); a.set_title(ti,loc='left',fontsize=10)
axs[0].set_xlim(5,33); axs[1].set_xlim(13,24.5)
h=[Line2D([],[],ls='',marker='o',color=GRN,ms=8,label='sized by the rule (b = b*)'),Line2D([],[],ls='',marker='s',color=RED,ms=8,label='full-range sizing at PR = 128'),
   Line2D([],[],ls='',marker='o',color=GRY,ms=8,label='other accuracy-safe points')]
f.legend(handles=h,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.06),fontsize=9)
f.tight_layout(rect=(0,0.06,1,1)); f.savefig(out+'fig6_s6.png',bbox_inches='tight'); plt.close(f)
print('ok')

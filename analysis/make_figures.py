"""Rebuild the data figures of the manuscript from the CSVs in results/.
Usage: python analysis/make_figures.py [results_dir] [out_dir]
Fig. 1 (flow diagram), Fig. 3 (histograms), Fig. 5 (per-layer) and Fig. 10 (stuck-at)
are produced by the campaign scripts and are not rebuilt here."""
import sys, csv, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
R = sys.argv[1] if len(sys.argv) > 1 else 'results/full_rerun_v1'
out = (sys.argv[2] if len(sys.argv) > 2 else 'figures') + '/'
os.makedirs(out, exist_ok=True)
rd = lambda f: list(csv.DictReader(open(os.path.join(R, f))))
P2, P3, P4, P5, P5b, P6 = (rd(x) for x in ['P2_clip_grid.csv','P3_c10_full_acc.csv','P4_c100_full_acc.csv','P5_vgg8.csv','P5b_vgg8_full.csv','P6_hw_energy.csv'])
g = lambda rows, **kv: [r for r in rows if all(int(float(r[k])) == v for k, v in kv.items())]
one = lambda rows, **kv: g(rows, **kv)[0]
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.color':'#e6e6e6','grid.linewidth':0.6,'axes.axisbelow':True,'savefig.dpi':300})
RED='#a83232'; GRN='#2e6b26'; GRY='#8994a6'; NAV='#34506e'; INK='#333333'
FP10=90.04; FP100=72.93
def save(f,n): f.savefig(out+n,bbox_inches='tight'); plt.close(f)

# ---------- Fig 2
a5=sorted(g(P2,adc=5),key=lambda r:int(r['pr'])); pr=[int(r['pr']) for r in a5]; acc5=[float(r['acc']) for r in a5]
full={p:float(one(P3,pr=p,adc=5)['acc']) for p in (64,96,128)}
f,(a,b)=plt.subplots(1,2,figsize=(9.2,3.6),gridspec_kw={'width_ratios':[2.2,1]})
x=np.arange(len(pr))
a.axvspan(-0.5,3.5,color=GRN,alpha=0.08,lw=0); a.axvspan(3.5,5.5,color='#d9a227',alpha=0.12,lw=0); a.axvspan(5.5,7.5,color=RED,alpha=0.10,lw=0)
for xx,t,c in [(1.5,'accuracy-safe',GRN),(4.5,'soft loss','#8a6a10'),(6.5,'collapse',RED)]: a.text(xx,99,t,ha='center',va='top',color=c)
a.axhline(FP10,ls=':',color=INK,lw=1.2)
a.plot(x,acc5,'-o',color=RED,lw=2,ms=5,label='N = 256 clipping grid')
for p,v in full.items():
    i=pr.index(p); a.plot(i,v,'o',mfc='none',mec=RED,mew=2,ms=11)
    a.annotate(f'{v:.2f}',(i,v),xytext=(0,-17) if p==64 else (14,8),textcoords='offset points',ha='center',color=RED,fontsize=9)
a.set_xticks(x,pr); a.set_xlim(-0.5,7.5); a.set_ylim(0,104); a.set_xlabel('Parallel read, PR (rows)'); a.set_ylabel('Top-1 accuracy (%)')
a.set_title('(a) 5-bit ADC across PR',loc='left',fontsize=10)
bs=[5,6,7,8]; bv=[float(one(P3,pr=128,adc=b)['acc']) for b in bs]; cols=[RED,GRN,GRY,GRY]
b.bar(range(4),bv,color=cols,edgecolor=INK,lw=0.6,width=0.62)
for i,v in enumerate(bv): b.text(i,v+1.5,f'{v:.2f}',ha='center',fontsize=9,fontweight='bold' if i==1 else None)
b.axhline(FP10,ls=':',color=INK,lw=1.2); b.set_xticks(range(4),bs); b.set_ylim(0,104)
b.set_xlabel('ADC resolution, b (bit)'); b.set_title('(b) PR = 128 across ADC resolution',loc='left',fontsize=10)
h=[Line2D([],[],color=RED,marker='o',lw=2,label='N = 256 clipping grid'),Line2D([],[],ls='',marker='o',mfc='none',mec=RED,mew=2,ms=9,label='full 10 000-image test'),
   Patch(fc=GRN,ec=INK,label='minimum-viable resolution'),Line2D([],[],ls=':',color=INK,label=f'floating point ({FP10:.2f}%)')]
f.legend(handles=h,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(0.5,-0.06),fontsize=9)
f.tight_layout(rect=(0,0.06,1,1)); save(f,'fig2.png')

# ---------- Fig 3
kind={(64,5):'rec',(128,6):'rec',(128,7):'dom',(128,8):'dom',(64,8):'dom',(128,5):'rej'}
pts=[(f'PR{p}/{b}-bit',float(one(P6,pr=p,adc=b)['tops_w']),float(one(P6,pr=p,adc=b)['area'])/1e8,float(one(P3,pr=p,adc=b)['acc']),k) for (p,b),k in kind.items()]
f,a=plt.subplots(figsize=(6.2,4.4))
a.plot([28.00,26.98],[3.05,3.41],'--',color=GRN,lw=1.5,zorder=1)
off={'PR64/5-bit':(0,-30),'PR128/6-bit':(-8,14),'PR128/7-bit':(8,8),'PR128/8-bit':(8,6),'PR64/8-bit':(8,-22),'PR128/5-bit':(0,14)}
for n,t,ar,ac,k in pts:
    if k=='rej': a.scatter(t,ar,marker='X',s=140,color=RED,zorder=3,edgecolor='white',lw=1)
    else: a.scatter(t,ar,s=80,color=GRN if k=='rec' else GRY,zorder=3,edgecolor='white',lw=1)
    lab=f'{n}\n{t:.2f} TOPS/W'+(f', {ac:.2f}%' if k!='dom' else '')+(' (reject)' if k=='rej' else '')
    dx,dy=off[n]; a.annotate(lab,(t,ar),xytext=(dx,dy),textcoords='offset points',fontsize=8.5,
       ha='right' if n=='PR128/6-bit' else ('center' if n in('PR64/5-bit','PR128/5-bit') else 'left'),color=INK)
a.set_xlim(5,31); a.set_ylim(2.6,5.2); a.set_xlabel('Energy efficiency (TOPS/W)'); a.set_ylabel('Chip area (×10$^8$ µm$^2$)')
h=[Line2D([],[],ls='',marker='o',color=GRN,ms=8,label='recommended (frontier)'),Line2D([],[],ls='',marker='o',color=GRY,ms=8,label='accuracy-safe, dominated'),
   Line2D([],[],ls='',marker='X',color=RED,ms=9,label='rejected on accuracy'),Line2D([],[],ls='--',color=GRN,label='frontier')]
a.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.14),ncol=2,frameon=False,fontsize=9)
save(f,'fig3.png')

# ---------- Fig 4 (two panels, no dual axis)
bb=['5','6','7','8','12']
# 12-bit point: earlier matched sweep (TOPS/W 1.17, area 3.04e9 um^2), not in P6
tw=[float(one(P6,pr=128,adc=b)['tops_w']) for b in (5,6,7,8)]+[1.17]; ar=[float(one(P6,pr=128,adc=b)['area'])/1e8 for b in (5,6,7,8)]+[30.4]; cc=[RED,GRN,GRY,GRY,GRY]
f,(a,b)=plt.subplots(1,2,figsize=(8.6,3.4))
a.bar(range(5),tw,color=cc,edgecolor=INK,lw=0.6,width=0.62)
for i,v in enumerate(tw): a.text(i,v+0.5,f'{v:.2f}',ha='center',fontsize=9,fontweight='bold' if i==1 else None)
a.set_xticks(range(5),bb); a.set_xlabel('ADC resolution, b (bit)'); a.set_ylabel('Energy efficiency (TOPS/W)'); a.set_ylim(0,31)
a.set_title('(a) Efficiency at PR = 128',loc='left',fontsize=10)
b.bar(range(5),ar,color=cc,edgecolor=INK,lw=0.6,width=0.62); b.set_yscale('log')
for i,v in enumerate(ar): b.text(i,v*1.08,f'{v:.2f}' if v<10 else f'{v:.1f}',ha='center',fontsize=9)
b.set_xticks(range(5),bb); b.set_xlabel('ADC resolution, b (bit)'); b.set_ylabel('Chip area (×10$^8$ µm$^2$, log)'); b.set_ylim(1,60)
b.set_title('(b) Chip area at PR = 128',loc='left',fontsize=10)
h=[Patch(fc=GRN,ec=INK,label='minimum viable'),Patch(fc=RED,ec=INK,label='rejected on accuracy'),Patch(fc=GRY,ec=INK,label='accuracy-safe, dominated')]
f.legend(handles=h,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.05),fontsize=9)
f.tight_layout(rect=(0,0.06,1,1)); save(f,'fig4.png')

# ---------- Fig 5
lab=['PR64\n5-bit','PR128\n5-bit','PR128\n6-bit','PR128\n8-bit']; v=[float(one(P4,pr=p,adc=b)['acc']) for p,b in ((64,5),(128,5),(128,6),(128,8))]; c=[GRY,RED,GRN,GRY]
f,a=plt.subplots(figsize=(5.8,3.6))
a.bar(range(4),v,color=c,edgecolor=INK,lw=0.6,width=0.55); a.axhline(FP100,ls=':',color=INK,lw=1.2)
a.text(-0.4,FP100+6,f'floating point {FP100:.2f}%',fontsize=9,color=INK)
for i,y in enumerate(v): a.text(i,y+1.5,f'{y:.2f}',ha='center',fontsize=9,fontweight='bold' if i==2 else None)
a.set_xticks(range(4),lab); a.set_ylim(0,90); a.set_ylabel('Top-1 accuracy (%)')
h=[Patch(fc=GRN,ec=INK,label='minimum-viable resolution'),Patch(fc=GRY,ec=INK,label='accuracy-safe'),Patch(fc=RED,ec=INK,label='collapse')]
a.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.18),ncol=3,frameon=False,fontsize=9)
save(f,'fig5.png')

# ---------- Fig 7 (stacked panels, shared x, no dual axis)
clip5=[float(r['clip_frac']) for r in a5]
f,(a,b)=plt.subplots(2,1,figsize=(6.2,5.6),sharex=True,gridspec_kw={'height_ratios':[1.15,1]})
floor=1e-9
xs=pr; y=[max(c,floor) for c in clip5]
a.axhspan(2e-5,4.09e-4,color='#d9a227',alpha=0.15,lw=0)
a.axhline(0.1,ls='--',color=INK,lw=1)
a.text(17,0.13,'≈ 0.1: collapse',fontsize=8.5,color=INK)
a.text(17,6e-5,'safe/soft boundary',fontsize=8.5,color='#7a5c0a')
a.plot(xs[1:],y[1:],'-o',color=RED,lw=2,ms=5,label='5-bit ADC')
a.plot(xs[0]-1.2,floor,'o',mfc='white',mec=RED,mew=1.5,ms=6)
a.plot([17.2,32,64,96,128],[floor]*5,'s',mfc='white',mec=NAV,mew=1.5,ms=6,label='8-bit ADC (no clipped code)')
a.set_yscale('log'); a.set_ylim(5e-10,3); a.set_ylabel(r'Clipped fraction, $f_{\mathrm{clip}}$')
a.set_title('(a) Pre-ADC clipped fraction (open markers: zero, drawn at 10$^{-9}$)',loc='left',fontsize=10)
a.legend(loc='lower right',frameon=False,fontsize=9)
b.axhline(FP10,ls=':',color=INK,lw=1.2)
b.plot(xs,acc5,'-o',color=RED,lw=2,ms=5,label='5-bit ADC')
a8=sorted(g(P2,adc=8),key=lambda r:int(r['pr']))
b.plot([int(r['pr']) for r in a8],[float(r['acc']) for r in a8],'-s',color=NAV,lw=1.5,ms=5,label='8-bit ADC')
b.set_ylim(0,100); b.set_ylabel('Top-1 accuracy (%)'); b.set_xlabel('Parallel read, PR (rows)')
b.set_title('(b) Accuracy, same runs (N = 256)',loc='left',fontsize=10); b.legend(loc='lower left',frameon=False,fontsize=9)
b.set_xticks(pr)
f.tight_layout(); save(f,'fig7.png')

# ---------- Fig 9
cfg=[(64,5),(64,8),(128,5),(128,6),(128,8)]; lab=[f'PR{p}\n{b}-bit' for p,b in cfg]
src=[(g(P5b,pr=p,adc=b) or g(P5,pr=p,adc=b))[0] for p,b in cfg]
v=[float(r['acc']) for r in src]; cl=[f"{float(r['clip_frac']):.3g}" for r in src]
probe=[0 if r in P5b else 1 for r in src]; c=[GRY,GRY,RED,GRN,GRY]
f,a=plt.subplots(figsize=(6.4,3.8))
for i in range(5):
    a.bar(i,v[i],color=c[i],edgecolor=INK,lw=0.6,width=0.55,hatch='//' if probe[i] else None)
    a.text(i,v[i]+1.5,f'{v[i]:.2f}',ha='center',fontsize=9,fontweight='bold')
    a.text(i,v[i]+7,f'clip {cl[i]}',ha='center',fontsize=8,color='#555')
a.set_xticks(range(5),lab); a.set_ylim(0,108); a.set_ylabel('Top-1 accuracy (%)')
h=[Patch(fc=GRN,ec=INK,label='minimum-viable resolution'),Patch(fc=GRY,ec=INK,label='accuracy-safe'),Patch(fc=RED,ec=INK,label='collapse'),Patch(fc='white',ec=INK,hatch='//',label='512-image probe')]
a.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.18),ncol=4,frameon=False,fontsize=8.5)
save(f,'fig9.png')
print('ok')

# ---------- Fig 11: per-layer converter allocation at PR = 128 (S3)
S3p = os.path.join(R, 'S3_mixed_precision.csv')
if os.path.exists(S3p):
    S3 = [r for r in csv.DictReader(open(S3p)) if r['n_images'] == '1024']
    byp = {r['plan']: r for r in S3}
    f, a = plt.subplots(figsize=(5.8, 3.8))
    a.axhline(FP10, ls=':', color=INK, lw=1.2)
    a.text(0.01, FP10 + 2, f'floating point {FP10:.2f}%', fontsize=8.5, color=INK)
    for key, col, mk, lab in [('share', NAV, 'o', 'ranked by share of clipped values, $s_\\ell$'),
                              ('rate', '#d9a227', 's', 'ranked by clipped fraction, $c_\\ell$')]:
        pts = [byp['uniform5']] + [byp[f'{key}_top{k}'] for k in (4, 8, 12, 16)] + [byp['uniform6']]
        xs = [100 * float(r['conv_share_6bit']) for r in pts]; ys = [float(r['acc']) for r in pts]
        a.plot(xs, ys, '-' + mk, color=col, lw=1.8, ms=6, label=lab, zorder=3)
    for k in (4, 8, 16):
        r = byp[f'share_top{k}']
        a.annotate(f'{k}', (100 * float(r['conv_share_6bit']), float(r['acc'])), xytext=(0, 8),
                   textcoords='offset points', ha='center', fontsize=8, color=NAV)
    r12 = byp['share_top12']
    a.annotate('12', (100 * float(r12['conv_share_6bit']), float(r12['acc'])), xytext=(0, -15), textcoords='offset points', ha='center', fontsize=8, color=NAV)
    r16s, r16r = byp['share_top16'], byp['rate_top16']
    a.annotate(f"{float(r16s['acc']):.1f}%", (100 * float(r16s['conv_share_6bit']), float(r16s['acc'])),
               xytext=(-40, -6), textcoords='offset points', fontsize=8.5, color=NAV)
    a.annotate(f"{float(r16r['acc']):.1f}%", (100 * float(r16r['conv_share_6bit']), float(r16r['acc'])),
               xytext=(8, -14), textcoords='offset points', fontsize=8.5, color='#8a6a10')
    a.set_xlim(-3, 103); a.set_ylim(0, 100)
    a.text(2, 31, 'numbers: layers raised to 6 bits', fontsize=8, color='#555')
    a.set_xlabel('Conversions at 6 bits (% of all, rest at 5 bits)'); a.set_ylabel('Top-1 accuracy (%)')
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False, fontsize=8.5)
    save(f, 'fig11.png')

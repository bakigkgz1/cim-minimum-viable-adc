import os
UC = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else 'analysis/unit_column'
exec(open(os.path.join(UC, 'paper.py')).read())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.color':'#e6e6e6','grid.linewidth':0.6,'axes.axisbelow':True,'savefig.dpi':300})
RED='#a83232'; GRN='#2e6b26'; GRY='#8994a6'; NAV='#34506e'; INK='#333333'; GOLD='#c08a1e'; PUR='#4b2a7b'
out=str(OUT)+'/'
NET=[(('resnet18','cifar10'),NAV,'o','ResNet18/CIFAR-10'),(('vgg8','cifar10'),GOLD,'^','VGG8/CIFAR-10'),(('resnet18','cifar100'),RED,'D','ResNet18/CIFAR-100'),(('vgg8','cifar100'),PUR,'v','VGG8/CIFAR-100')]
def ylog(a):
    a.set_yscale('symlog',linthresh=1); a.set_ylim(-100,1.2)
    a.set_yticks([0,-0.5,-1,-10,-100]); a.set_yticklabels(['0','−0.5','−1','−10','−100'])

# ================= Fig 1: datapath schematic =================
def box(a,x,y,w,h,txt,fc='white',ec=INK,fs=8.5,lw=1.0,tc=INK,**k):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.06',fc=fc,ec=ec,lw=lw))
    a.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=tc,**k)
def arrow(a,x0,y0,x1,y1,col=INK,lw=1.0):
    a.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='-|>',color=col,lw=lw,shrinkA=0,shrinkB=0,mutation_scale=8))
f,axs=plt.subplots(1,2,figsize=(11.6,5.2))
for a,mode in zip(axs,('analog','digital')):
    a.set_xlim(0,10.5); a.set_ylim(-0.4,11); a.axis('off')
    a.set_title('(a) Offset removed with an analog unit column' if mode=='analog' else '(b) Digital offset removal',loc='left',fontsize=10.5)
    rows=[8.0,7.2,6.4,5.6,4.4]
    for i,y in enumerate(rows):
        a.text(0.2,y,f'$x_{{{i+1}}}$' if i<4 else '$x_{PR}$',fontsize=8.5,va='center')
        a.plot([0.75,5.75 if mode=='analog' else 4.55],[y,y],color=GRY,lw=0.8)
    a.text(0.32,5.0,'·\n·',fontsize=8,va='center',linespacing=0.6)
    a.text(0.05,9.0,'PR rows read\ntogether, one\ninput bit each',fontsize=8.4,color=INK,va='center')
    cols=[1.5,2.3,3.1,3.9]
    for j,x in enumerate(cols):
        a.plot([x,x],[3.9,8.5],color=NAV,lw=1.2)
        for i,y in enumerate(rows):
            on=((i*3+j*5)%4)!=0
            a.add_patch(Rectangle((x-0.13,y-0.13),0.26,0.26,fc=NAV if on else 'white',ec=NAV,lw=0.8))
        box(a,x-0.33,2.75,0.66,0.7,'ADC',fc='#e8edf3',ec=NAV,fs=8.0)
        arrow(a,x,3.9,x,3.45,NAV)
    a.text(2.7,9.55,'data columns (weight bits $w_{ij}$)\n$y_j=\\sum_i w_{ij}x_i$',fontsize=8.6,ha='center',va='center',color=NAV)
    if mode=='analog':
        x=5.2
        a.plot([x,x],[3.9,8.5],color=RED,lw=1.6)
        for y in rows: a.add_patch(Rectangle((x-0.13,y-0.13),0.26,0.26,fc=RED,ec=RED,lw=0.8))
        box(a,x-0.33,2.75,0.66,0.7,'ADC',fc='#f6e3e3',ec=RED,fs=8.0)
        arrow(a,x,3.9,x,3.45,RED)
        a.text(6.0,9.55,'unit column (all cells on)\n$u=\\sum_i x_i\\geq y_j$ for every $j$',fontsize=8.6,ha='left',va='center',color=RED)
        a.text(5.75,3.1,'clips first,\nonce $u\\geq 2^b$',fontsize=8.4,color=RED,va='center')
        box(a,0.9,0.15,6.6,1.25,'shift-add over bit planes and partitions, then\nout$_j$ = Σ$y_j$ − shift·Σ$Q(u)$ − offset row + const',fc='#f5f5f5',fs=8.4)
        for xx in cols+[x]: arrow(a,xx,2.75,xx,1.4,GRY,0.8)
        a.text(7.75,0.78,'a clipped $u$\nis subtracted\nfrom every\noutput $j$',fontsize=8.6,color=RED,va='center')
    else:
        box(a,5.3,6.2,2.0,1.6,'popcount\n$u=\\sum_i x_i$\n(exact, digital)',fc='#e7f0e4',ec=GRN,fs=8.4)
        arrow(a,4.55,7.0,5.3,7.0,GRN)
        box(a,7.7,6.2,2.5,1.6,'offset row\nshift·$\\sum_i w_{ij}$\nprecomputed\nfrom the weights',fc='#e7f0e4',ec=GRN,fs=8.2)
        box(a,0.9,0.15,6.6,1.25,'shift-add over bit planes and partitions, then\nout$_j$ = Σ$y_j$ − shift·Σ$u$ − offset row + const',fc='#f5f5f5',fs=8.4)
        for xx in cols: arrow(a,xx,2.75,xx,1.4,GRY,0.8)
        arrow(a,6.3,6.2,6.3,1.4,GRN,0.9); arrow(a,8.95,6.2,7.3,1.4,GRN,0.9)
        a.text(6.45,4.2,'converters see\nonly data\ncolumns;\noffset terms\nare exact',fontsize=8.6,color=GRN,va='center')
f.tight_layout(); f.savefig(out+'fig1_s9.png',bbox_inches='tight'); plt.close(f)

# ================= Fig 2: accuracy vs PR, unit column vs digital offset =================
f,(a,b)=plt.subplots(1,2,figsize=(10.6,4.0))
for key,col,mk,lab in NET[:3]:
    nrm=sorted([p for p in PTS if (p['model'],p['dataset'])==key and p['seed']==1234 and p['b']==5],key=lambda p:p['pr'])
    a.plot([p['pr'] for p in nrm],[p['d'] for p in nrm],'--',marker=mk,color=col,mfc='white',ms=5.5,lw=1.1)
    ex=sorted([d for d in EX if (d['model'],d['dataset'])==key and d['sub']==128 and d['b']==5 and d['seed']==1234 and d['ck']=='orig'],key=lambda d:d['pr'])
    a.plot([d['pr'] for d in ex],[d['d'] for d in ex],'-',marker=mk,color=col,ms=6,lw=1.6)
    nr=sorted([d for d in N256 if (d['model'],d['dataset'])==key and d['b']==6],key=lambda d:d['pr'])
    b.plot([d['pr'] for d in nr],[d['d'] for d in nr],'--',marker=mk,color=col,mfc='white',ms=5.5,lw=1.1)
    ex=sorted([d for d in EX if (d['model'],d['dataset'])==key and d['sub']==256 and d['b']==6 and d['seed']==1234 and d['ck']=='orig'],key=lambda d:d['pr'])
    b.plot([d['pr'] for d in ex],[d['d'] for d in ex],'-',marker=mk,color=col,ms=6,lw=1.6)
for ax,t in ((a,'(a) 128-row array, 5-bit ADC'),(b,'(b) 256-row array, 6-bit ADC')):
    ylog(ax); ax.axhline(-TOL,color=INK,ls=':',lw=1); ax.set_xlabel('Parallel read PR (rows)'); ax.set_title(t,loc='left',fontsize=10)
a.set_ylabel('Accuracy change vs lossless (pp)'); a.set_xticks([32,48,64,80,96,112,128]); b.set_xticks([128,160,192,224,256])
h=[Line2D([],[],color=c,marker=m,ls='',ms=6,label=l) for _,c,m,l in NET[:3]]+[Line2D([],[],color=INK,ls='--',marker='o',mfc='white',ms=5,label='analog unit column'),Line2D([],[],color=INK,ls='-',marker='o',ms=5,label='offset converted exactly')]
f.legend(handles=h,loc='lower center',ncol=5,frameon=False,fontsize=8.4,bbox_to_anchor=(0.5,-0.04))
f.tight_layout(rect=(0,0.07,1,1)); f.savefig(out+'fig2_s9.png',bbox_inches='tight'); plt.close(f)

# ================= Fig 3: per-conversion clipping, and dose-response =================
f,(a,b)=plt.subplots(1,2,figsize=(11.0,4.3),gridspec_kw={'width_ratios':[1,1.25]})
for key,col,mk,lab in NET:
    pre=f'{key[0]}|{key[1]}|'+('v100|' if key==('vgg8','cifar100') else '')
    prs=sorted({int(k[len(pre)+7:].split('|')[0][2:]) for k in OLX if k.startswith(pre+'sub128|') and k.count('|')==pre.count('|')+1})
    a.plot(prs,[max(tail(f'{pre}sub128|pr{p}',5,'signal'),1e-9) for p in prs],'-',marker=mk,color=col,ms=5,lw=1.4)
    a.plot(prs,[tail(f'{pre}sub128|pr{p}',5,'offset') for p in prs],':',marker=mk,color=col,mfc='white',ms=5,lw=1.4)
a.set_yscale('log'); a.set_ylim(1e-8,1); a.set_xlabel('Parallel read PR (rows), 128-row array'); a.set_ylabel('Share of conversions ≥ 32 (5-bit full scale)')
a.set_title('(a) Clipping per conversion, open loop',loc='left',fontsize=10)
a.legend(handles=[Line2D([],[],color=INK,ls='-',marker='o',ms=5,label='data columns'),Line2D([],[],color=INK,ls=':',marker='o',mfc='white',ms=5,label='offset terms (unit column, offset row)')],frameon=False,fontsize=8,loc='lower right')
# (b) normal points: total tail; exact points: signal tail
b.axvline(1e-5,color=GRY,ls='--',lw=1); b.axhline(-TOL,color=INK,ls=':',lw=1)
for key,col,mk,lab in NET:
    for p in PTS:
        if (p['model'],p['dataset'])!=key: continue
        b.errorbar(max(p['f'],2e-9),p['d'],yerr=[[p['d']-p['lo']],[p['hi']-p['d']]],fmt=mk,color=col,mfc='white',mec=col,ms=4.5,capsize=1.5,lw=0.7,alpha=0.8)
    for d in N256:
        if (d['model'],d['dataset'])!=key: continue
        b.errorbar(max(d['f_all'],2e-9),d['d'],yerr=[[d['d']-d['lo']],[d['hi']-d['d']]],fmt=mk,color=col,mfc='white',mec=col,ms=4.5,capsize=1.5,lw=0.7,alpha=0.8)
    for d in A10:
        if (d['model'],d['dataset'])!=key: continue
        b.errorbar(max(d['f_all'],2e-9),d['d'],yerr=[[d['d']-d['lo']],[d['hi']-d['d']]],fmt=mk,color=col,mfc='white',mec=col,ms=4.5,capsize=1.5,lw=0.7,alpha=0.8)
    for d in EX:
        if (d['model'],d['dataset'])!=key: continue
        b.errorbar(max(d['f_sig'],2e-9),d['d'],yerr=[[d['d']-d['lo']],[d['hi']-d['d']]],fmt=mk,color=col,mec=col,ms=6,capsize=1.5,lw=1.0)
b.set_xscale('log'); b.set_xlim(1e-9,1e-1); ylog(b)
b.set_xlabel('Open-loop clipped share: all conversions (open symbols)\nor data columns only, offset exact (filled)'); b.set_ylabel('Accuracy change vs lossless (pp)')
b.set_title('(b) Accuracy against the clipping that remains',loc='left',fontsize=10)
b.text(1.15e-5,0.5,'10$^{-5}$',fontsize=8,color=GRY)
h=[Line2D([],[],color=c,marker=m,ls='',ms=6,label=l) for _,c,m,l in NET]
f.legend(handles=h,loc='lower center',ncol=4,frameon=False,fontsize=8.4,bbox_to_anchor=(0.5,-0.05))
f.tight_layout(rect=(0,0.06,1,1)); f.savefig(out+'fig3_s9.png',bbox_inches='tight'); plt.close(f)

# ================= Fig 4: feedback with and without the analog offset =================
L7={}
for r in csv.DictReader(open(R/'S7_layers.csv')): L7.setdefault(r['exp'],[]).append(r)
L8={}
for r in csv.DictReader(open(R/'S8_layers.csv')): L8.setdefault(r['exp'],[]).append(r)
nrm=sorted(L7['L_pr80_b5_all'],key=lambda r:int(r['index'])); exa=sorted(L8['R_pr80_all_exempt'],key=lambda r:int(r['index']))
x=list(range(1,len(nrm)+1))
f,(a,b)=plt.subplots(1,2,figsize=(10.6,3.8))
a.plot(x,[max(float(r['open_ge']),1e-10) for r in nrm],':o',color=GRY,mfc='white',ms=4,lw=1.2,label='open loop')
a.plot(x,[max(float(r['closed_ge']),1e-10) for r in nrm],'-o',color=RED,ms=4,lw=1.5,label='in network, analog unit column')
a.plot(x,[max(float(r['closed_ge']),1e-10) for r in exa],'-s',color=GRN,ms=4,lw=1.5,label='in network, offset exact')
a.set_yscale('log'); a.set_ylim(5e-11,1e-1); a.set_xlabel('CIM layer (network order)'); a.set_ylabel('Share of partial sums ≥ 32')
a.set_title('(a) Clipping by layer, PR80/5-bit',loc='left',fontsize=10); a.legend(fontsize=7.8,frameon=False,loc='lower left')
b.plot(x,[100*(float(r['closed_mean'])/float(r['open_mean'])-1) for r in nrm],'-o',color=RED,ms=4,lw=1.5,label='analog unit column')
b.plot(x,[100*(float(r['closed_mean'])/float(r['open_mean'])-1) for r in exa],'-s',color=GRN,ms=4,lw=1.5,label='offset exact')
b.axhline(0,color=INK,lw=0.8); b.set_xlabel('CIM layer (network order)'); b.set_ylabel('Mean partial sum vs open loop (%)')
b.set_title('(b) Mean partial sum of each layer',loc='left',fontsize=10); a.set_xticks(range(2,21,2)); b.set_xticks(range(2,21,2)); b.legend(fontsize=8,frameon=False,loc='upper left')
f.tight_layout(); f.savefig(out+'fig4_s9.png',bbox_inches='tight'); plt.close(f)
print('max |shift| exact %.2f%%, normal max %.1f%%'%(max(abs(100*(float(r['closed_mean'])/float(r['open_mean'])-1)) for r in exa),max(100*(float(r['closed_mean'])/float(r['open_mean'])-1) for r in nrm)))

# ================= Fig 5: hardware =================
HW={}
for fn in (R/'S8_hw.csv', R/'S9_hw.csv'):
    for r in csv.DictReader(open(fn)):
        HW[(r['arch'],int(r['sub_array']),int(r['pr']),int(r['adc']),'dig' if 'digoff' in r['tag'] else 'ana')]=r
json.dump({'|'.join(map(str,k)):v for k,v in HW.items()},open(OUT/'hw.json','w'),indent=1)
PLAN=[(128,128,[(5,'dig','digital offset',GRN,'*'),(6,'ana','criterion τ = 10$^{-5}$',NAV,'o'),(7,'ana','practical full range',GOLD,'D'),(8,'ana','strict full range',RED,'s')]),
      (256,256,[(6,'dig','digital offset',GRN,'*'),(7,'ana','criterion τ = 10$^{-5}$',NAV,'o'),(8,'ana','practical full range',GOLD,'D'),(9,'ana','strict full range',RED,'s')])]
f,axs=plt.subplots(1,2,figsize=(10.6,4.1))
for a,arch in zip(axs,('MLSA','SAR')):
    for sub,pr,pts in PLAN:
        xs=[];ys=[]
        for bb,src,lab,col,mk in pts:
            r=HW[(arch,sub,pr,bb,src)]; xx=float(r['area'])/1e8; yy=float(r['tops_w'])
            xs.append(xx); ys.append(yy)
            a.plot(xx,yy,mk,color=col,ms=11 if mk=='*' else 7,mec='white' if mk!='*' else col,zorder=3)
            a.annotate(f'{bb}-bit',(xx,yy),xytext=(5,4),textcoords='offset points',fontsize=8.6,color=INK)
        a.plot(xs,ys,'-',color=GRY,lw=0.8,zorder=1)
        a.annotate('PR128, 128-row array' if sub==128 else 'PR256, 256-row array',(xs[0],ys[0]),xytext=(-6,-16) if sub==128 else (0,14),textcoords='offset points',fontsize=7.8,color=INK,ha='left' if sub==128 else 'center',va='top' if sub==128 else 'bottom')
    a.margins(x=0.08,y=0.12)
    a.set_xlabel('Chip area (10$^{8}$ µm$^{2}$)'); a.set_ylabel('Energy efficiency (TOPS/W)')
    a.set_title(f'({"a" if arch=="MLSA" else "b"}) {"flash-type MLSA" if arch=="MLSA" else "SAR"} converter',loc='left',fontsize=10)
h=[Line2D([],[],color=c,marker=m,ls='',ms=9 if m=='*' else 6,label=l) for _,_,l,c,m in PLAN[0][2]]
f.legend(handles=h,loc='lower center',ncol=4,frameon=False,fontsize=8.4,bbox_to_anchor=(0.5,-0.04))
f.tight_layout(rect=(0,0.07,1,1)); f.savefig(out+'fig5_s9.png',bbox_inches='tight'); plt.close(f)
print('figs ok')

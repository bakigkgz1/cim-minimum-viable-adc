import os
UC = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else 'analysis/unit_column'
exec(open(os.path.join(UC, 'paper.py')).read())
import math
def sci(x, d=1):
    if x == 0: return '0'
    e = int(math.floor(math.log10(abs(x)))); m = x / 10**e
    if round(m, d) >= 10: m /= 10; e += 1
    return f'{m:.{d}f}×10^{{−{-e}}}' if e < 0 else f'{m:.{d}f}×10^{{{e}}}'
def pp(x): return ('+' if x >= 0 else '−') + f'{abs(x):.2f}'
def ci(lo, hi): return 'none changed' if abs(lo) < 1e-9 and abs(hi) < 1e-9 else f'[{pp(lo)}, {pp(hi)}]'
def nimg(n): return f'{n:,}'.replace(',', ' ')
NM = {('resnet18','cifar10'):'ResNet18/C10',('vgg8','cifar10'):'VGG8/C10',('resnet18','cifar100'):'ResNet18/C100',('vgg8','cifar100'):'VGG8/C100'}
def ana(m, d, sub, pr, b):
    if sub == 128:
        c = [p for p in PTS if (p['model'],p['dataset'],p['pr'],p['b'],p['seed']) == (m,d,pr,b,1234)] + [x for x in A10 if (x['model'],x['dataset'],x['pr'],x['b']) == (m,d,pr,b)]
    else:
        c = [x for x in N256 if (x['model'],x['dataset'],x['pr'],x['b']) == (m,d,pr,b)]
    return max(c, key=lambda p: p['n']) if c else None
def exa(m, d, sub, pr, b, seed=1234, ck='orig'):
    c = [x for x in EX if (x['model'],x['dataset'],x['sub'],x['pr'],x['b'],x['seed'],x['ck']) == (m,d,sub,pr,b,seed,ck) or (x['model'],x['dataset'],x['sub'],x['pr'],x['b'],x['ck']) == (m,d,sub,pr,b,'v100')]
    return c[0] if c else None
T = {}
# ---- mechanism table
rows = []
for m, d, sub, pr, b in [('resnet18','cifar10',128,72,5),('resnet18','cifar10',128,80,5),('resnet18','cifar10',128,96,5),('resnet18','cifar10',128,128,5),
                         ('vgg8','cifar10',128,80,5),('vgg8','cifar10',128,128,5),('resnet18','cifar100',128,72,5),('resnet18','cifar100',128,128,5),('vgg8','cifar100',128,96,5),
                         ('resnet18','cifar10',256,192,6),('resnet18','cifar10',256,256,6),('vgg8','cifar10',256,256,6),('resnet18','cifar100',256,256,6)]:
    cfg = f'{m}|{d}|' + ('v100|' if (m,d)==('vgg8','cifar100') else '') + f'sub{sub}|pr{pr}'
    o = OLX[cfg+'|offset']; a = OLX[cfg]
    ps = tail(cfg, b, 'signal'); po = tail(cfg, b, 'offset')
    sh = int(o[f'ge_{2**b}']) / int(a[f'ge_{2**b}'])
    A = ana(m, d, sub, pr, b); E = exa(m, d, sub, pr, b)
    rows.append([NM[(m,d)], str(sub), f'{pr}/{b}', f"{100*int(o['conversions'])/int(a['conversions']):.2f}%", sci(ps), sci(po), f'{po/ps:.0f}', f'{100*sh:.0f}%',
                 pp(A['d']) + ('' if A['n'] == 10000 else ('^{a}' if A['n'] == 2048 else '^{b}')), pp(E['d']) + ' ' + ci(E['lo'], E['hi']) + ('' if E['n'] == 10000 else '^{b}')])
T['mech'] = rows
# ---- all exact-offset points (supplement)
rows = []
for x in sorted(EX, key=lambda x: (['resnet18|cifar10','vgg8|cifar10','resnet18|cifar100','vgg8|cifar100'].index(x['model']+'|'+x['dataset']), x['sub'], x['b'], x['pr'], x['ck'], x['seed'])):
    rows.append([NM[(x['model'],x['dataset'])], str(x['sub']), str(x['pr']), str(x['b']), str(x['seed']) + ({'ck2':', ckpt 2','ck3':', ckpt 3'}.get(x['ck'], '')), nimg(x['n']), sci(x['f_sig']), sci(x['f_all']), pp(x['d']), ci(x['lo'], x['hi']), verdict(x['lo'], x['hi'])])
T['ex'] = rows
# ---- 256-row unit-column points (supplement)
T['n256'] = [[NM[(x['model'],x['dataset'])] + ({'ck2':', ckpt 2','ck3':', ckpt 3'}.get(x.get('ck',''), '')), str(x['sub']), str(x['pr']), str(x['b']), nimg(x['n']), sci(x['f_all']) if x['f_all'] else '0', pp(x['d']), ci(x['lo'], x['hi']), verdict(x['lo'], x['hi'])]
             for x in sorted(N256, key=lambda x: (x['model'], x['dataset'], x['b'], x['pr'])) + sorted(A10, key=lambda x: (x['model'], x['dataset'], x['ck'], x['pr'], x['b']))]
# ---- sizing table
def bm(cfg, part, tau): return next(b for b in range(2, 11) if tail(cfg, b, part) <= tau)
MEAS = {  # measured minimum resolution; i = inferred from a measured neighbour, a–b = lower value inconclusive
 ('resnet18','cifar10',128,64):('5–6^{i}','4–5^{i}'), ('resnet18','cifar10',128,96):('6','5'), ('resnet18','cifar10',128,128):('6','5'),
 ('resnet18','cifar10',256,160):('6–7^{i}','5–6'), ('resnet18','cifar10',256,192):('7^{i}','6'), ('resnet18','cifar10',256,256):('7','6'),
 ('vgg8','cifar10',128,64):('5','4'), ('vgg8','cifar10',128,96):('6^{i}','5'), ('vgg8','cifar10',128,128):('6','5'),
 ('vgg8','cifar10',256,160):('—','5'), ('vgg8','cifar10',256,192):('—','5–6^{i}'), ('vgg8','cifar10',256,256):('7','6'),
 ('resnet18','cifar100',128,64):('6^{i}','5^{i}'), ('resnet18','cifar100',128,96):('6^{i}','5'), ('resnet18','cifar100',128,128):('6','6^{i}'),
 ('resnet18','cifar100',256,160):('—','≤7^{i}'), ('resnet18','cifar100',256,192):('—','≤7^{i}'), ('resnet18','cifar100',256,256):('7','7^{i}'),
 ('vgg8','cifar100',128,64):('≥6','5'), ('vgg8','cifar100',128,96):('≥6','5–6'), ('vgg8','cifar100',128,128):('7','6')}
rows = []
for (m, d, sub, pr), (ma, me) in MEAS.items():
    cfg = f'{m}|{d}|' + ('v100|' if (m,d)==('vgg8','cifar100') else '') + f'sub{sub}|pr{pr}'
    rows.append([NM[(m,d)], str(sub), str(pr), str(math.ceil(math.log2(pr+1))), str(math.ceil(math.log2(pr))), str(bm(cfg,'all',1e-5)), ma, str(bm(cfg,'signal',5e-5)), me])
T['size'] = rows
# ---- hardware
HW = json.load(open(OUT/'hw.json'))
def hw(arch, sub, pr, b, src): return HW[f'{arch}|{sub}|{pr}|{b}|{src}']
rows = []
for sub, pr, pts in [(128,64,[(4,'dig','digital offset','VGG8/C10'),(5,'dig','digital offset','all four^{b}'),(6,'ana','practical full range; criterion for the C100 pairs','all four')]),
                     (128,128,[(5,'dig','digital offset','C10 pairs'),(6,'ana','criterion τ = 10^{−5}','all but VGG8/C100'),(7,'ana','practical full range; criterion for VGG8/C100','all four'),(8,'ana','strict full range','all four')]),
                     (256,256,[(6,'dig','digital offset','C10 pairs'),(7,'ana','criterion; also ResNet18/C100 with digital offset','all measured^{c}'),(8,'ana','practical full range','all measured^{c}'),(9,'ana','strict full range','all measured^{c}')])]:
    for b, src, lab, ok in pts:
        r1 = hw('MLSA', sub, pr, b, src); r2 = hw('SAR', sub, pr, b, src)
        E1 = float(r1['energy_pJ']); A1 = float(r1['area'])
        rows.append([f'{sub}/{pr}', str(b), lab, ok, f"{float(r1['tops_w']):.2f}", f"{A1/1e8:.2f}", f"{100*float(r1['adc_E_pJ'])/E1:.0f}% / {100*float(r1['adc_area'])/A1:.0f}%",
                     f"{float(r1['fps']):,.0f}".replace(',', ' '), f"{float(r2['tops_w']):.2f}", f"{float(r2['area'])/1e8:.2f}", f"{float(r2['fps']):,.0f}".replace(',', ' ')])
T['hw'] = rows
json.dump(T, open(OUT/'tables.json', 'w'), indent=1, ensure_ascii=False)
for k, v in T.items():
    print('==', k)
    for r in v: print(r)

import os
UC = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else 'analysis/unit_column'
exec(open(os.path.join(UC, 'all.py')).read())
import json, csv
W = R
# ---------------- open loop (all / signal / offset) from S8 and S9
OLX = {}
for f in (R/'S8_openloop.csv', R/'S9_openloop.csv'):
    for r in csv.DictReader(open(f)): OLX[r['cfg']] = r
def tail(cfg, b, part='all'):
    k = cfg + ('' if part == 'all' else '|' + part); r = OLX[k]
    return int(r[f'ge_{2**b}']) / int(r['conversions'])
NAMES = {('resnet18','cifar10'):'ResNet18/C10',('vgg8','cifar10'):'VGG8/C10',('resnet18','cifar100'):'ResNet18/C100'}
def stat(c, ref):
    return cmp(c, ref)
def verdict(lo, hi):
    return 'safe' if lo > -TOL else ('lossy' if hi < -TOL else 'inconcl.')
# ---------------- exact-offset (digital offset) points and 256-row unit-column points
EX = []; N256 = []; ABL = []
def load_runs(path, pdir):
    for r in csv.DictReader(open(path)):
        yield r, np.load(pdir/r['pred_file'])
seedref = {}
for r, c in load_runs(R/'S9_runs.csv', R/'S9_preds'):
    if r['bits'] == '8': seedref[(r['phase'], r['model'])] = c
for path, pdir in ((R/'S8_runs.csv', R/'S8_preds'), (R/'S9_runs.csv', R/'S9_preds')):
    for r, c in load_runs(path, pdir):
        key = (r['model'], r['dataset']); sub = int(r['sub_array']); pr = int(r['pr']); b = int(r['bits']); ex = int(r['exempt_dummy'])
        if r['phase'] == 'F' or b == 8: continue
        seed = 1234 if r['phase'] in 'RAB' else int(r['phase'][1:])
        ref = REF[key] if seed == 1234 else seedref[(r['phase'], r['model'])]
        s = stat(c, ref)
        cfg = f"{key[0]}|{key[1]}|sub{sub}|pr{pr}" + ('' if seed == 1234 else f'|seed{seed}')
        d = dict(model=key[0], dataset=key[1], sub=sub, pr=pr, b=b, seed=seed, ex=ex, **s,
                 f_all=tail(cfg, b), f_sig=tail(cfg, b, 'signal'), f_off=tail(cfg, b, 'offset'))
        if ex == 1: EX.append(d)
        elif ex == 0 and sub == 256: N256.append(d)
        else: ABL.append(d)
# normal 128-row points (S6/S7, 41 points) with signal tail where S8/S9 open loop exists
for p in PTS:
    cfg = f"{p['model']}|{p['dataset']}|sub128|pr{p['pr']}" + ('' if p['seed'] == 1234 else f"|seed{p['seed']}")
    p['f_sig'] = tail(cfg, p['b'], 'signal') if cfg in OLX else None
# ---------------- S10: independent checkpoints, VGG8/CIFAR-100, denser CIFAR-100 points, noise
A10 = []; NOISE = []
for r in csv.DictReader(open(R/'S10_openloop.csv')): OLX[r['cfg']] = r
for r in csv.DictReader(open(R/'S11_openloop.csv')): OLX[r['cfg']] = r
_r10 = list(csv.DictReader(open(R/'S10_runs.csv')))
_ref10 = {(r['model'], r['dataset'], r['ckpt']): np.load(R/'S10_preds'/r['pred_file']) for r in _r10 if r['bits'] == '8'}
NAMES[('vgg8','cifar100')] = 'VGG8/C100'
for r in _r10:
    if r['bits'] == '8': continue
    c = np.load(R/'S10_preds'/r['pred_file']); key = (r['model'], r['dataset']); ck = r['ckpt']
    ref = _ref10.get((r['model'], r['dataset'], ck), REF.get(key))
    s = stat(c, ref); sub = int(r['sub_array']); pr = int(r['pr']); b = int(r['bits']); ex = int(r['exempt_dummy'])
    cfg = f"{key[0]}|{key[1]}|{ck}|sub{sub}|pr{pr}"
    d = dict(model=key[0], dataset=key[1], sub=sub, pr=pr, b=b, seed=1234, ex=ex, ck=ck, sigma=float(r['sigma_lsb']), **s,
             f_all=tail(cfg, b), f_sig=tail(cfg, b, 'signal'), f_off=tail(cfg, b, 'offset'))
    if r['phase'] == 'N': NOISE.append(d)
    elif ex == 1: EX.append(d)
    else: A10.append(d)
for d in EX: d.setdefault('ck', 'orig')
# ---------------- S11: VGG8/CIFAR-100 and ResNet18 4-bit boundary on the full test set
_r11 = list(csv.DictReader(open(R/'S11_runs.csv')))
_ref11 = {(r['model'], r['dataset'], r['ckpt']): np.load(R/'S11_preds'/r['pred_file']) for r in _r11 if r['bits'] == '8'}
for r in _r11:
    if r['bits'] == '8': continue
    c = np.load(R/'S11_preds'/r['pred_file']); key = (r['model'], r['dataset']); ck = r['ckpt']
    ref = _ref11.get((r['model'], r['dataset'], ck), REF.get(key))
    s = stat(c, ref); sub = int(r['sub_array']); pr = int(r['pr']); b = int(r['bits']); ex = int(r['exempt_dummy'])
    cfg = f"{key[0]}|{key[1]}|{ck}|sub{sub}|pr{pr}"
    d = dict(model=key[0], dataset=key[1], sub=sub, pr=pr, b=b, seed=1234, ex=ex, ck=ck, sigma=0.0, **s,
             f_all=tail(cfg, b), f_sig=tail(cfg, b, 'signal'), f_off=tail(cfg, b, 'offset'))
    same = lambda x: (x['model'], x['dataset'], x['sub'], x['pr'], x['b'], x.get('ck', 'orig'), x['seed']) == (d['model'], d['dataset'], sub, pr, b, ck, 1234)
    if ex == 1:
        EX[:] = [x for x in EX if not same(x)]; EX.append(d)
    else:
        A10[:] = [x for x in A10 if not same(x)]; A10.append(d)
# ---------------- S14: VGG8/CIFAR-10 256-row PR192/5-bit with exact offset on the full test set (replaces the 2 048-image S9 point)
if (R/'S14_runs.csv').exists():
    for r in csv.DictReader(open(R/'S14_runs.csv')):
        if r['differential'] != '0' or r['model'] != 'vgg8': continue
        c = np.load(R/'S14_preds'/r['pred_file']); key = (r['model'], r['dataset']); sub = int(r['sub_array']); pr = int(r['pr']); b = int(r['bits'])
        cfg = f"{key[0]}|{key[1]}|sub{sub}|pr{pr}"
        d = dict(model=key[0], dataset=key[1], sub=sub, pr=pr, b=b, seed=1234, ex=1, ck='orig', **stat(c, REF[key]),
                 f_all=tail(cfg, b), f_sig=tail(cfg, b, 'signal'), f_off=tail(cfg, b, 'offset'))
        EX[:] = [x for x in EX if (x['model'], x['dataset'], x['sub'], x['pr'], x['b'], x.get('ck', 'orig'), x['seed']) != (key[0], key[1], sub, pr, b, 'orig', 1234)]; EX.append(d)
json.dump(dict(EX=EX, N256=N256, ABL=ABL, A10=A10, NOISE=NOISE), open(OUT/'points.json', 'w'), indent=1, default=float)
if __name__ == '__main__':
    for d in sorted(EX, key=lambda d: (d['model'], d['dataset'], d['sub'], d['b'], d['pr'], d['seed'])):
        print(f"EX {NAMES[(d['model'],d['dataset'])]:14} {d['sub']} PR{d['pr']}/{d['b']} s{d['seed']} n={d['n']} d={d['d']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}] {verdict(d['lo'],d['hi'])} fs={d['f_sig']:.1e} fa={d['f_all']:.1e}")
    for d in N256 + ABL:
        print(f"N  {NAMES[(d['model'],d['dataset'])]:14} {d['sub']} PR{d['pr']}/{d['b']} ex{d['ex']} n={d['n']} d={d['d']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}] {verdict(d['lo'],d['hi'])} fa={d['f_all']:.1e}")

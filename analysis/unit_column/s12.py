# Converter errors (Tables 5-6), differential mappings (Section 4.8, Tables S17-S18) from S12-S17.
# Usage (repo root): python analysis/unit_column/s12.py   [P3_RESULTS=... P3_OUT=...]
import os
UC = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else 'analysis/unit_column'
exec(open(os.path.join(UC, 'paper.py')).read())
import csv, json, numpy as np
OUT = Path(os.environ.get('P3_OUT', 'figures')); OUT.mkdir(parents=True, exist_ok=True)
runs = list(csv.DictReader(open(R/'S12_runs.csv')))
OL = {r['cfg']: r for r in csv.DictReader(open(R/'S12_openloop.csv'))}
# ideal-converter counterparts of the converter-error runs
IDEAL = {('128', '5', '1'): R/'S8_preds'/'R_resnet18_cifar10_sub128_pr128_b5_ex1_n0.0_nb-1.npy',
         ('128', '5', '4'): R/'S8_preds'/'R_resnet18_cifar10_sub128_pr128_b5_ex1_n0.0_nb-1.npy',
         ('256', '6', '1'): R/'S8_preds'/'A_resnet18_cifar10_sub256_pr256_b6_ex1_n0.0_nb-1.npy',
         ('128', '6', '0'): P/'A_resnet18_cifar10_pr128_b6_s0_halfup_clamp0_n0.0_nb-1.npy'}
res = dict(errors=[], differential=[])
for r in runs:
    c = np.load(R/'S12_preds'/r['pred_file'])
    if r['differential'] == '0':
        s = cmp(c, np.load(IDEAL[(r['sub_array'], r['bits'], r['exempt_dummy'])]))
        res['errors'].append(dict(sub=int(r['sub_array']), b=int(r['bits']), exempt=int(r['exempt_dummy']), s=float(r['adc_err_lsb']),
                                  code_change=float(r['code_change_frac']), **s))
    else:
        s = cmp(c, REF[(r['model'], r['dataset'])])
        cfg = f"{r['model']}|{r['dataset']}|orig|sub{r['sub_array']}|pr{r['pr']}|diff"; o = OL[cfg]
        res['differential'].append(dict(model=r['model'], sub=int(r['sub_array']), pr=int(r['pr']), b=int(r['bits']), s=float(r['adc_err_lsb']),
                                        open_loop=int(o[f"ge_{2**int(r['bits'])}"]) / int(o['conversions']), **s))
# first-order cost of the differential design (Table S18)
H9 = {x['tag']: x for x in csv.DictReader(open(R/'S9_hw.csv'))}
H13 = {x['tag']: x for x in csv.DictReader(open(R/'S13_hw.csv'))}
res['cost'] = {}
for arch in ('MLSA', 'SAR'):
    for sub, b in ((128, 5), (256, 6)):
        x = H9[f'S9_digoff_{arch}_sub{sub}_PR{sub}_ADC{b}']; xa = H9[f'S9_digoff_{arch}_sub{sub}_PR{sub}_ADC{b+1}']
        y = H13[f'S13_digoff_{arch}_sub{sub}_PR{sub}_ADC{b-1}']
        E, Ea, Ec, Eo, T = (float(x[k]) for k in ('energy_pJ', 'adc_E_pJ', 'accum_E_pJ', 'other_E_pJ', 'tops_w'))
        ratio = float(y['adc_E_pJ']) / Ea; Ead = 2 * ratio * Ea   # the estimator charges M conversions for offset encoding
        res['cost'][f'{arch}|{sub}'] = dict(exact_offset=T, analog_unit_column=float(xa['tops_w']), ratio=ratio, adc_energy_factor=Ead / Ea,
                                           differential_best=T * E / (E - Ea + Ead), differential_worst=T * E / (2 * (Ec + Eo) + Ead))
# S14/S15: single error sources, gain/INL, second seed, full tests, analog (signed) differential
IDEAL14 = {('128', '5'): R/'S8_preds'/'R_resnet18_cifar10_sub128_pr128_b5_ex1_n0.0_nb-1.npy',
           ('256', '6'): R/'S8_preds'/'A_resnet18_cifar10_sub256_pr256_b6_ex1_n0.0_nb-1.npy'}
res['s14_s15'] = []
R11 = {(r['model'], r['dataset'], r['ckpt']): np.load(R/'S11_preds'/r['pred_file']) for r in csv.DictReader(open(R/'S11_runs.csv')) if r['bits'] == '8'}
for tag in ('S14', 'S15', 'S16'):
    for r in csv.DictReader(open(R/f'{tag}_runs.csv')):
        c = np.load(R/f'{tag}_preds'/r['pred_file'])
        ref = (np.load(IDEAL14[(r['sub_array'], r['bits'])]) if (r['differential'] == '0' and r['model'] == 'resnet18')
               else R11[(r['model'], r['dataset'], r['ckpt'])] if r['ckpt'] == 'v100' else REF[(r['model'], r['dataset'])])
        s = cmp(c, ref)
        res['s14_s15'].append(dict(run=tag, model=r['model'], sub=int(r['sub_array']), pr=int(r['pr']), b=int(r['bits']), differential=int(r['differential']),
                                   offset=float(r['err_offset']), transition=float(r['err_transition']), noise=float(r['err_noise']),
                                   gain_fs=float(r['err_gain_fs']), inl=float(r['err_inl']), seed=int(r['err_seed']), mismatch=float(r.get('mismatch') or 0), cols_over_half=r.get('cols_over_half', ''),
                                   against='ideal converter' if r['differential'] == '0' and r['model'] == 'resnet18' else '8-bit reference',
                                   code_change=float(r['code_change_frac']), altered=float(r['altered_frac']), **s))
for k, v in res['cost'].items():
    x = H9[f"S9_digoff_{k.split('|')[0]}_sub{k.split('|')[1]}_PR{k.split('|')[1]}_ADC{5 if k.endswith('128') else 6}"]
    E, Ea, Eo, T = (float(x[q]) for q in ('energy_pJ', 'adc_E_pJ', 'other_E_pJ', 'tops_w')); M = int(k.split('|')[1])
    Es = v['ratio'] * Ea
    v['signed_best'] = T * E / (E - Ea + Es); v['signed_worst'] = T * E / (E - Ea + Es + Eo)
# S17: conductance path (on/off ratio 10, device variation, read noise), ResNet18/CIFAR-10 PR128
if (R/'S17_runs.csv').exists():
    P17 = {(r['phase'], int(r['bits'])): np.load(R/'S17_preds'/r['pred_file']) for r in csv.DictReader(open(R/'S17_runs.csv'))}
    res['s17'] = {f'{c}|{b}': dict(vs_own_8bit=cmp(P17[(c, b)], P17[(c, 8)]), vs_ideal_cells=cmp(P17[(c, b)], P17[('C0', b)]))
                  for c in ('C0', 'C1', 'C2', 'C3') for b in (8, 5, 6)}
json.dump(res, open(OUT/'s12.json', 'w'), indent=1, default=float)
for d in res['errors']: print('ERR', d['sub'], d['b'], d['exempt'], d['s'], d['n'], f"{d['code_change']:.1e}", f"{d['d']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}]")
for d in res['differential']: print('DIFF', d['model'], d['sub'], d['b'], d['s'], d['n'], f"{d['open_loop']:.1e}", f"{d['d']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}]")
for k, v in res['cost'].items(): print('COST', k, {a: round(b, 3) for a, b in v.items()})
for d in res['s14_s15']: print(d['run'], d['model'], d['sub'], d['b'], 'd%d' % d['differential'], 'o%g t%g n%g g%g inl%g seed%d' % (d['offset'], d['transition'], d['noise'], d['gain_fs'], d['inl'], d['seed']), d['n'], f"{d['d']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}]", d['against'])

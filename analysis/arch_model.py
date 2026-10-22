"""First-order converter-architecture re-costing (manuscript Section 3.4 / Table VIII).

Every accuracy-safe operating point is normalized to PR64/5-bit. Conversions per inference
scale as ceil(H/PR); energy per conversion follows a SAR (2^b), flash (2^b - 1) or
noise-limited (4^b) law; the NeuroSim column is the measured converter dynamic energy
from P6_hw_energy.csv (matched levelOutput, multilevel sense amplifier).

Usage: python analysis/arch_model.py [results_dir]
"""
import csv, math, os, sys

R = sys.argv[1] if len(sys.argv) > 1 else "results/full_rerun_v1"
H = 128
P6 = {(int(r["pr"]), int(r["adc"])): r for r in csv.DictReader(open(os.path.join(R, "P6_hw_energy.csv")))}
P3 = {(int(r["pr"]), int(r["adc"])): r for r in csv.DictReader(open(os.path.join(R, "P3_c10_full_acc.csv")))}


def osr(b, loop):
    """OSR for a 1-bit-quantizer delta-sigma loop of order `loop` to reach a b-bit SQNR."""
    target = 6.02 * b + 1.76
    q = 6.02 + 1.76 - 10 * math.log10(math.pi ** (2 * loop) / (2 * loop + 1))
    return 10 ** ((target - q) / ((2 * loop + 1) * 10))


points = [(64, 5), (128, 6), (128, 7), (128, 8), (64, 8), (128, 12)]
ref = (64, 5)
nconv = lambda pr: math.ceil(H / pr)
laws = {"SAR": lambda b: 2 ** b, "flash": lambda b: 2 ** b - 1, "noise": lambda b: 4 ** b}
norm = lambda f, p: nconv(p[0]) * f(p[1]) / (nconv(ref[0]) * f(ref[1]))

hdr = ["point", "acc_%", "NeuroSim_E", "SAR_E", "flash_E", "noise_E", "area_SAR", "area_flash",
       "cycles_SAR", "cycles_flash", "OSR_L1", "OSR_L2"]
print(",".join(hdr))
e0 = float(P6[ref]["adc_E_pJ"])
for p in points:
    pr, b = p
    ns = f"{float(P6[p]['adc_E_pJ']) / e0:.2f}" if p in P6 else "-"
    acc = P3[p]["acc"] if p in P3 else "-"
    row = [f"PR{pr}/{b}-bit", acc, ns,
           f"{norm(laws['SAR'], p):.2f}", f"{norm(laws['flash'], p):.2f}", f"{norm(laws['noise'], p):.1f}",
           f"{2 ** b / 2 ** 5:.1f}", f"{(2 ** b - 1) / 31:.1f}",
           f"{nconv(pr) * b / (nconv(64) * 5):.2f}", f"{nconv(pr) / nconv(64):.1f}",
           f"{osr(b, 1):.1f}", f"{osr(b, 2):.1f}"]
    print(",".join(row))

print("\nmeasured PR64->128 converter-energy ratio at 8 bits:",
      round(float(P6[(128, 8)]["adc_E_pJ"]) / float(P6[(64, 8)]["adc_E_pJ"]), 3), "(model 0.5)")

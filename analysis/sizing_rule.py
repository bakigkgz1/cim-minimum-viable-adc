"""Clipping-tail sizing rule (manuscript Sections 3.9 and 4.5, Table VI, Fig. 5).

Reads the open-loop partial-sum histograms (S5_partial_sum_hist.csv, one unclipped pass of
256 training images per PR) and
  * computes the open-loop clipped fraction f_clip(PR, b) = P(y >= 2^b), Eq. (12),
  * sizes the ADC as b* = min{b : f_clip <= tau}, tau = 2e-5, Eq. (13),
  * repeats the sizing with the two-moment beta-binomial form, Eq. (14), and with the
    independent-bit binomial model, and compares with the full-range rule, Eq. (15),
  * checks the rule against the 33 measured operating points,
  * draws Fig. 5 if matplotlib is available.

Usage:  python analysis/sizing_rule.py [results_dir] [figure_dir]
"""
import csv, sys
from math import lgamma, log, exp
from pathlib import Path
import numpy as np

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results/full_rerun_v1")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
TAU = 2e-5
BITS = range(1, 10)

# ---------------------------------------------------------------- histograms
D = {}
with open(R / "S5_partial_sum_hist.csv") as f:
    for r in csv.DictReader(f):
        h = np.array([int(x) for x in r["hist"].split(";")], dtype=float)
        D.setdefault((r["model"], r["dataset"], int(r["pr"])), {})[r["layer"]] = h


def agg(cfg):
    hs = list(D[cfg].values()); n = max(len(h) for h in hs)
    return sum(np.pad(h, (0, n - len(h))) for h in hs)


def f_open(cfg, b):
    """Open-loop clipped fraction: share of partial sums at or above full scale 2^b."""
    h = agg(cfg); v = np.arange(len(h))
    return h[v >= 2 ** b].sum() / h.sum()


def lbeta(x, y): return lgamma(x) + lgamma(y) - lgamma(x + y)


def model_tail(cfg, b, kind):
    """Per-layer moment fit, conversion-weighted; kind = 'betabin' or 'binom'."""
    P = N = 0.0
    for h in D[cfg].values():
        n = len(h) - 1; v = np.arange(len(h)); w = h.sum()
        mu = (v * h).sum() / w; var = ((v - mu) ** 2 * h).sum() / w; p = mu / n
        thr = 2 ** b
        if thr > n:
            t = 0.0
        elif kind == "binom":
            t = sum(exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1 - p))
                    for k in range(thr, n + 1))
        else:
            rho = max((var / (n * p * (1 - p)) - 1) / (n - 1), 1e-9); s = 1 / rho - 1; a, bb = p * s, (1 - p) * s
            t = sum(exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + lbeta(k + a, n - k + bb) - lbeta(a, bb))
                    for k in range(thr, n + 1))
        P += t * w; N += w
    return P / N


def bstar(fn):
    return next(b for b in BITS if fn(b) <= TAU)


def b_fullrange(pr): return int(np.ceil(np.log2(pr + 1)))


# ---------------------------------------------------------------- Table VI
NAME = {"resnet18": "ResNet18", "vgg8": "VGG8"}; DS = {"cifar10": "C10", "cifar100": "C100"}
print(f"Table VI — open-loop clipped fraction and minimum resolution, tau = {TAU:g}")
print(f"{'network':14s} {'PR':>4s} {'f5':>9s} {'f6':>9s} {'b*hist':>6s} {'b*BB':>5s} {'b*bin':>5s} {'b_FR':>4s}")
for cfg in sorted(D, key=lambda c: (c[0] != "resnet18" or c[1] != "cifar10", c[0], c[1], c[2])):
    m, d, pr = cfg
    print(f"{NAME[m] + '/' + DS[d]:14s} {pr:4d} {f_open(cfg, 5):9.1e} {f_open(cfg, 6):9.1e} "
          f"{bstar(lambda b: f_open(cfg, b)):6d} {bstar(lambda b: model_tail(cfg, b, 'betabin')):5d} "
          f"{bstar(lambda b: model_tail(cfg, b, 'binom')):5d} {b_fullrange(pr):4d}")

# ---------------------------------------------------------------- validation, 33 measured points
# (model, dataset, PR, b, top-1 %, test images) from P2_clip_grid (N=256), P3/P4 full tests (N=10 000),
# P1/P2 locks (N=512) and P5/P5b (VGG8); see the README table for the files.
FP = {("resnet18", "cifar10"): 90.04, ("vgg8", "cifar10"): 90.04, ("resnet18", "cifar100"): 72.93}
M = [("resnet18", "cifar10", pr, 5, a, 256) for pr, a in
     {16: 91.41, 32: 91.41, 64: 90.63, 72: 91.80, 80: 83.98, 88: 69.53, 96: 25.39, 128: 11.33}.items()]
for pr in (16, 32, 64, 96, 128):
    M += [("resnet18", "cifar10", pr, b, 90.82, 512) for b in (6, 8, 12)]
M += [("resnet18", "cifar10", 128, 7, 90.04, 10000),
      ("vgg8", "cifar10", 64, 5, 91.80, 512), ("vgg8", "cifar10", 64, 8, 91.80, 512),
      ("vgg8", "cifar10", 128, 5, 34.81, 10000), ("vgg8", "cifar10", 128, 6, 90.10, 10000),
      ("vgg8", "cifar10", 128, 8, 91.80, 512),
      ("resnet18", "cifar100", 64, 5, 72.41, 10000), ("resnet18", "cifar100", 128, 5, 1.00, 10000),
      ("resnet18", "cifar100", 128, 6, 72.93, 10000), ("resnet18", "cifar100", 128, 8, 72.70, 10000)]
ok = 0
for m, d, pr, b, a, n in M:
    f = f_open((m, d, pr), b) if 2 ** b <= pr else 0.0
    se = 100 * (0.9 * 0.1 / n) ** 0.5
    safe = a >= FP[(m, d)] - (2 * max(se, 0.3) if (m, d) == ("resnet18", "cifar10") else max(2 * se, 1.0))
    ok += (f <= TAU) == safe
    if (f <= TAU) != safe:
        print("MISMATCH", m, d, pr, b, f, a)
print(f"\nrule classifies {ok}/{len(M)} measured operating points correctly")

# ---------------------------------------------------------------- Fig. 5
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300})
RED, GRN, GRY, NAV = "#a83232", "#2e6b26", "#8994a6", "#34506e"
prs = sorted(p for (m, d, p) in D if (m, d) == ("resnet18", "cifar10"))
c10 = lambda p: ("resnet18", "cifar10", p)
ol = [f_open(c10(p), 5) for p in prs]
meas = {}
with open(R / "P2_clip_grid.csv") as f:
    for r in csv.DictReader(f):
        if (r["model"], r["dataset"], r["adc"], r["hardware"]) == ("resnet18", "cifar10", "5", "0") and \
                float(r["read_noise"]) == 0 and float(r["rate_stuck_0"]) == 0 and float(r["rate_stuck_1"]) == 0:
            meas[int(r["pr"])] = float(r["clip_frac"])
fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.7))
a.axhline(TAU, ls="--", color=GRN, lw=1.2); a.text(17, TAU * 1.6, "τ = 2×10$^{-5}$", color=GRN, fontsize=8.5)
a.plot([p for p, v in zip(prs, ol) if v > 0], [v for v in ol if v > 0], "-o", color=NAV, lw=1.8, ms=5,
       label="open-loop (unclipped statistics)")
mp = sorted(p for p in meas if meas[p] > 0)
a.plot(mp, [meas[p] for p in mp], "-s", color=RED, lw=1.8, ms=5, label="measured (clipping network)")
a.set_yscale("log"); a.set_ylim(1e-8, 2); a.set_xticks([16, 32, 64, 96, 128])
a.set_xlabel("Parallel read, PR (rows)"); a.set_ylabel("Clipped fraction, 5-bit ADC")
a.set_title("(a) Open-loop prediction vs measurement", loc="left", fontsize=10)
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8.2)
emp = [bstar(lambda k: f_open(c10(p), k)) for p in prs]
bbp = [bstar(lambda k: model_tail(c10(p), k, "betabin")) for p in prs]
b.plot(prs, [b_fullrange(p) for p in prs], "-o", color=GRY, lw=1.6, ms=4, label="full-range rule ⌈log$_2$(PR+1)⌉")
b.plot(prs, [v + 0.06 for v in bbp], "--s", color="#d9a227", lw=1.6, ms=4, label="beta-binomial (mean, variance)")
b.plot(prs, emp, "-o", color=NAV, lw=2.0, ms=5, label="open-loop histogram, ResNet18/C10")
for m, d, mk, lab in [("vgg8", "cifar10", "^", "VGG8/C10"), ("resnet18", "cifar100", "D", "ResNet18/C100")]:
    xs = sorted(p for (mm, dd, p) in D if (mm, dd) == (m, d))
    ys = [bstar(lambda k: f_open((m, d, p), k)) for p in xs]
    b.plot([x + (2 if mk == "^" else -2) for x in xs], [y - 0.08 for y in ys], mk, color=RED, ms=6, ls="",
           label=lab + " (open-loop)")
b.set_yticks([4, 5, 6, 7, 8]); b.set_ylim(3.5, 8.5); b.set_xticks([16, 32, 64, 96, 128])
b.set_xlabel("Parallel read, PR (rows)"); b.set_ylabel("Minimum ADC resolution (bit)")
b.set_title("(b) Sizing chart, τ = 2×10$^{-5}$", loc="left", fontsize=10)
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=7.8)
OUT.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(OUT / "fig5_sizing_rule.png", bbox_inches="tight")
print("wrote", OUT / "fig5_sizing_rule.png")

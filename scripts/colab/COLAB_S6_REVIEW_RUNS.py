# === PAPER III — S6: REVIEWER-DRIVEN RUNS (one Colab cell, T4 GPU) ===
# Answers the independent review before resubmission. All runs use the functional path (hardware=0)
# and a TEXTBOOK b-bit converter: 2^b codes, output clamped to [0, 2^b - 1]. Under this convention a
# partial sum is altered exactly when y >= 2^b, i.e. the paper's clipped-fraction definition, and the
# accuracy model matches the hardware costing (L = 2^b levels).
#
#   A  ResNet18/CIFAR-10: re-measure the whole sizing boundary with the textbook converter, incl.
#      b*-1 (minimality: 4-bit at PR 16-32) and 6-bit at PR 80-128; 10 000 images near the boundary
#   B  held-out networks near the boundary: VGG8/C10 and ResNet18/C100, 5-bit at PR 64-88 and
#      6-bit at PR 128 on the full test set (threshold tau is NOT re-fitted)
#   C  coarse LSB (2 or 4 cells) with three rounding schemes: half-up (as in S4), half-even and
#      stochastic (unbiased) -> is the S4 collapse caused by rounding bias?
#   D  conversion noise with the lower clamp (as in NeuroSim/S1b) and symmetric noise (no lower
#      clamp, zero-mean error) -> is the 0.11-LSB limit caused by rectified noise?
# Every configuration also records the open-loop (unclipped) partial-sum histograms on 512 training
# images (4 batches, for a confidence interval) and on 256 test images, so the rule can be checked
# on the same images and at the new read widths. Per-image correctness is saved for paired tests.
#
# Approximate run time on a T4 (setup 0–8 min; resume-safe, finished rows are skipped, so the
# cell can be re-run after a disconnect and continues where it stopped):
#   reference runs (8-bit, 3 networks, 10 000 images)   ~ 35 min
#   A  ResNet18/C10 boundary (11 read widths)            ~ 2.7 h
#   C  coarse LSB, 12 runs x 1024 images                ~ 20 min
#   D  noise, 16 runs x 1024 images                     ~ 30 min
#   B  VGG8/C10 + ResNet18/C100                         ~ 2.5 h
#   Total ~ 6.5 h. Order: REF, A, C, D, B. Switch phases off below to split it across sessions.
# Outputs (Drive/NeuroSim15v_paper3/full_rerun_v1/):
#   S6_runs.csv        one row per accuracy run
#   S6_openloop.csv    open-loop tail counts per configuration, source (train batch k / test)
#   S6_hist.csv        per-layer open-loop histograms at the new read widths (S5 format)
#   S6_preds/*.npy     per-image correctness (for McNemar tests against the 8-bit reference)

RUN_REF, RUN_A, RUN_C, RUN_D, RUN_B = True, True, True, True, True
N_FULL, N_MID, N_QUICK = -1, 32, 16        # -1 = full 10 000 test images; 32 batches = 2048; 16 = 1024

import csv, os, re, shutil, subprocess, sys, time, types, hashlib, math
from datetime import datetime, timezone
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")

ROOT  = Path("/content/NeuroSim15v")
DRIVE = Path("/content/drive/MyDrive/NeuroSim15v_paper3")
OUT   = DRIVE / "full_rerun_v1"; LOG = OUT / "logs"
OUT.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
COMMIT = "9825ef4"

# ---------------------------------------------------------------- setup
if not ROOT.exists():
    subprocess.check_call(["git", "clone", "-b", "2DInferenceV1.5-dev",
                           "https://github.com/neurosim/NeuroSim.git", str(ROOT)])
    subprocess.check_call(["git", "-C", str(ROOT), "checkout", "-q", COMMIT])
print("NeuroSim HEAD:", subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip())
os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "pytorch-quantization"))
(ROOT / "models").mkdir(exist_ok=True)
for p in (DRIVE / "models").glob("*.pth"):
    if not (ROOT / "models" / p.name).exists():
        shutil.copy2(p, ROOT / "models" / p.name); print("copy model", p.name)
# NeuroSim's pytorch-quantization needs its compiled CUDA extension (cuda_ext).
# Its setup.py fails on current Colab, so the extension is compiled with torch's JIT
# builder instead and dropped into the package; a .pth file makes the package visible
# to this kernel and to the subprocesses used in S2.
import site
PQ = ROOT / "pytorch-quantization"
Path(site.getsitepackages()[0], "neurosim_pq.pth").write_text(str(PQ) + "\n")

def pq_ready():
    r = subprocess.run([sys.executable, "-c",
                        "import torch, pytorch_quantization.cuda_ext, pytorch_quantization.cim.modules.macro"],
                       cwd="/content", capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:])
    return r.returncode == 0

if pq_ready():
    print("pytorch-quantization ready")
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "absl-py", "prettytable", "scipy",
                    "pyyaml", "ninja"], check=True)
    print("Compiling cuda_ext with torch JIT (~3-6 min, first time only)...", flush=True)
    import torch, glob
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU in this runtime. Runtime > Change runtime type > T4 GPU, then run this cell again.")
    nvcc = sorted(glob.glob("/usr/local/cuda*/bin/nvcc")) or ([shutil.which("nvcc")] if shutil.which("nvcc") else [])
    if not nvcc:
        raise RuntimeError("GPU present but CUDA toolkit (nvcc) not found under /usr/local/cuda*. Send me the output of: !ls /usr/local; !nvidia-smi")
    cuda_home = str(Path(nvcc[-1]).resolve().parents[1])
    os.environ["CUDA_HOME"] = cuda_home
    import torch.utils.cpp_extension as cpp_ext
    cpp_ext.CUDA_HOME = cuda_home          # module caches CUDA_HOME at import time
    print("CUDA_HOME =", cuda_home, "| torch CUDA", torch.version.cuda, flush=True)
    from torch.utils.cpp_extension import load
    bdir = Path("/content/pq_build"); bdir.mkdir(exist_ok=True)
    load(name="cuda_ext", sources=[str(PQ / "src" / "tensor_quant.cpp"), str(PQ / "src" / "tensor_quant_gpu.cu")],
         build_directory=str(bdir), verbose=False)
    so = sorted(bdir.glob("cuda_ext*.so"))[0]
    shutil.copy2(so, PQ / "pytorch_quantization" / so.name)
    assert pq_ready(), "cuda_ext still not importable - send me the output above"
    print("pytorch-quantization ready (cuda_ext compiled)")

import torch, torch.nn as nn
import pytorch_quantization.cim.modules.macro as macro
from inference import parse_args
from quantize import quantize_model
from dataset import get_cifar10
from pytorch_quantization.utils import misc, make_path, hook
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)

utc = lambda: datetime.now(timezone.utc).isoformat()

def load_rows(path):
    if not path.exists(): return []
    with path.open() as f: return list(csv.DictReader(f))

def append_row(path, row):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new: w.writeheader()
        w.writerow(row)

def done(path, key, **kv):
    return any(all(str(r.get(k)) == str(v) for k, v in kv.items()) for r in load_rows(path)) if key else False


# ---------------------------------------------------------------- CIFAR-100 loader (Normalize 0.5, as trained)
import dataset as dataset_mod
def get_cifar100_05(batch_size, data_root="/tmp/public_dataset/pytorch", train=True, val=True, **kw):
    import torchvision.transforms as T, torchvision.datasets as DS
    from torch.utils.data import DataLoader
    root = os.path.expanduser(os.path.join(data_root, "cifar100-data")); kw.setdefault("num_workers", 2)
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]); ds = []
    if train: ds.append(DataLoader(DS.CIFAR100(root=root, train=True, download=True, transform=tf), batch_size=batch_size, shuffle=True, **kw))
    if val:   ds.append(DataLoader(DS.CIFAR100(root=root, train=False, download=True, transform=tf), batch_size=batch_size, shuffle=False, **kw))
    return ds[0] if len(ds) == 1 else ds
dataset_mod.get_cifar100 = get_cifar100_05

DS_CACHE = DRIVE / "datasets_cache"
def sync_datasets(to_drive=True):
    src, dst = (ROOT / "datasets", DS_CACHE) if to_drive else (DS_CACHE, ROOT / "datasets")
    if not src.exists(): return
    dst.mkdir(parents=True, exist_ok=True)
    for d in src.iterdir():                                # copy only dataset folders the other side lacks
        if d.is_dir() and not (dst / d.name).exists():
            print("dataset cache:", d.name, "->", dst, flush=True); shutil.copytree(d, dst / d.name)
sync_datasets(to_drive=False)                              # reuse CIFAR files cached on Drive, if any

import numpy as np
S6R = OUT / "S6_runs.csv"; S6O = OUT / "S6_openloop.csv"; S6H = OUT / "S6_hist.csv"
PRED = OUT / "S6_preds"; PRED.mkdir(exist_ok=True)
DONE_KEYS = {r["key"] for r in load_rows(S6R)}
OL_DONE = {(r["model"], r["dataset"], r["pr"]) for r in load_rows(S6O)}
H_DONE = {(r["model"], r["dataset"], r["pr"]) for r in load_rows(S6H)}
S5_DONE = {(r["model"], r["dataset"], r["pr"]) for r in load_rows(OUT / "S5_partial_sum_hist.csv")}


def build(model_name, dataset, pr, seed=1234):
    sys.argv = ["inference.py", "--dataset", dataset, "--model", model_name, "--hardware", "0", "--ppa", "0",
                "--num_batches", "16", "--batch_size", "64", "--calib_batch_size", "128", "--seed", str(seed),
                "--parallel_read", str(pr), "--adc_precision", "8", "--sub_array", "[128,128]",
                "--data_path", "./datasets/", "--model_path", "./models/", "--test_name", f"s6_{model_name}_{dataset}_{pr}"]
    args = parse_args()
    torch.cuda.set_device(args.gpu); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    args.data_path = args.data_path + args.dataset + "/"
    args.logdir = make_path.makepath(args, ["log_interval", "test_interval", "logdir", "epochs", "gpu",
                                            "ngpu", "debug", "data_path", "model_path"])
    misc.ensure_dir(args.logdir); misc.logger.init(args.logdir, "s6"); args.logger = misc.logger.info
    getter = get_cifar10 if dataset == "cifar10" else dataset_mod.get_cifar100
    dq = getter(args.calib_batch_size, args.data_path, train=True, val=False)
    dt = getter(args.batch_size, args.data_path, train=False, val=True)
    hook.make_records(args); args.hook = True; args.write_network = True
    model = quantize_model(args, nn.CrossEntropyLoss(), dq, dt).cuda().eval()

    # converter settings shared by every CIM layer, changed between runs
    C = dict(stats=False, bits=8, s=0, rounding="halfup", noise=0.0, noise_mode="clamp0", conv="textbook")
    hist, cnt = {}, {"nelem": 0, "nalt": 0}
    for name, m in model.named_modules():
        if isinstance(m, macro.CIM) and hasattr(m, "_adc_quant"):
            orig = m._adc_quant
            def wrapped(self, x, _orig=orig, _n=name):
                if self._adc_quantizer is None or self._adc_quantizer._disabled or self._adc_quantizer._if_calib:
                    return _orig(x)
                if C["stats"]:                                   # exact, unclipped partition sums
                    v = x.detach().round().clamp(0, pr).long().flatten()
                    h = torch.bincount(v, minlength=pr + 1).cpu()
                    hist[_n] = hist[_n] + h if _n in hist else h
                    return x
                L = float(2 ** C["s"]); y = x / L
                if C["rounding"] == "halfup":
                    q = torch.floor(y + 0.5)
                elif C["rounding"] == "halfeven":
                    q = torch.round(y)                           # round half to even: unbiased on integers
                else:
                    q = torch.floor(y + torch.rand_like(y))      # stochastic rounding: E[q] = y
                if C["noise"] > 0:
                    q = torch.round(q + C["noise"] * torch.randn_like(q))
                top = 2 ** C["bits"] - (1 if C["conv"] == "textbook" else 0)
                cnt["nelem"] += q.numel(); cnt["nalt"] += int((q > top).sum().item())
                q = q.clamp(max=top) if C["noise_mode"] == "symmetric" and C["noise"] > 0 else q.clamp(0, top)
                return q * L
            m._adc_quant = types.MethodType(wrapped, m)
    return model, dq, dt, C, hist, cnt


def openloop(model, dq, dt, C, hist, mname, dset, pr):
    """Histogram of exact partition sums: 4 training batches (512 images) and 256 test images."""
    rows = []
    def tail_row(source, h):
        v = torch.arange(h.numel())
        r = dict(model=mname, dataset=dset, pr=pr, source=source, conversions=int(h.sum()))
        for b in range(2, 9):
            r[f"ge_{2**b}"] = int(h[v >= 2 ** b].sum())
        return r
    C["stats"] = True
    with torch.no_grad():
        for i, (x, _) in enumerate(dq):
            if i >= 4: break
            hist.clear(); model(x.cuda())
            rows.append(tail_row(f"train{i}", sum(hist.values())))
            if i == 0: layer_hist = {k: v.clone() for k, v in hist.items()}
            else:
                for k, v in hist.items(): layer_hist[k] += v
        hist.clear()
        for i, (x, _) in enumerate(dt):
            if i >= 4: break
            model(x.cuda())
        rows.append(tail_row("test256", sum(hist.values())))
    C["stats"] = False; hist.clear()
    if (mname, dset, str(pr)) not in OL_DONE:
        for r in rows: append_row(S6O, r)
        OL_DONE.add((mname, dset, str(pr)))
    if (mname, dset, str(pr)) not in H_DONE and (mname, dset, str(pr)) not in S5_DONE:
        for n, h in layer_hist.items():
            hh = h[:pr + 1].tolist()
            append_row(S6H, dict(model=mname, dataset=dset, pr=pr, layer=n, conversions=int(sum(hh)),
                                 hist=";".join(str(int(v)) for v in hh)))
        H_DONE.add((mname, dset, str(pr)))
    tot = sum(r["conversions"] for r in rows[:4])
    return {b: sum(r[f"ge_{2**b}"] for r in rows[:4]) / tot for b in range(2, 9)}, \
           {b: rows[4][f"ge_{2**b}"] / rows[4]["conversions"] for b in range(2, 9)}


def run(phase, model, dt, C, cnt, mname, dset, pr, nb, bits, s=0, rounding="halfup", noise=0.0,
        noise_mode="clamp0", ol=None, seed=1234):
    key = f"{phase}|{mname}|{dset}|pr{pr}|b{bits}|s{s}|{rounding}|{noise_mode}|n{noise}|nb{nb}"
    if key in DONE_KEYS:
        print("skip", key); return
    C.update(bits=bits, s=s, rounding=rounding, noise=noise, noise_mode=noise_mode, conv="textbook")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cnt["nelem"] = cnt["nalt"] = 0; t0 = time.time(); correct = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dt):
            if nb >= 0 and i >= nb: break
            correct.append((model(x.cuda()).argmax(1) == y.cuda()).cpu().numpy())
    correct = np.concatenate(correct); n = correct.size
    fname = re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".npy"; np.save(PRED / fname, correct)
    olt, ols = ol if ol else ({}, {})
    row = dict(ts=utc(), key=key, phase=phase, model=mname, dataset=dset, pr=pr, bits=bits, conv="textbook",
               lsb_cells=2 ** s, rounding=rounding, noise_mode=noise_mode, sigma_lsb=noise, n_images=n,
               acc=round(100.0 * correct.mean(), 4), altered_frac=f"{cnt['nalt'] / max(cnt['nelem'], 1):.3e}",
               openloop_train=f"{olt.get(bits, float('nan')):.3e}" if s == 0 else "",
               openloop_test=f"{ols.get(bits, float('nan')):.3e}" if s == 0 else "",
               seconds=round(time.time() - t0, 1), pred_file=fname)
    append_row(S6R, row); DONE_KEYS.add(key)
    print(f"S6 {phase} {mname}/{dset} PR{pr} {bits}b LSB{2**s} {rounding} {noise_mode} σ={noise}: "
          f"acc {row['acc']} (n={n}) altered {row['altered_frac']} open-loop {row['openloop_train']} "
          f"[{row['seconds']}s]", flush=True)


def campaign(mname, dset, jobs):
    """jobs: list of (phase, pr, bits, nb, extra-kwargs). One model build per read width."""
    for pr in sorted({j[1] for j in jobs}):
        todo = [j for j in jobs if j[1] == pr and
                f"{j[0]}|{mname}|{dset}|pr{pr}|b{j[2]}|s{j[4].get('s', 0)}|{j[4].get('rounding', 'halfup')}|"
                f"{j[4].get('noise_mode', 'clamp0')}|n{j[4].get('noise', 0.0)}|nb{j[3]}" not in DONE_KEYS]
        if not todo:
            print(f"skip {mname}/{dset} PR{pr} (all done)"); continue
        t0 = time.time()
        model, dq, dt, C, hist, cnt = build(mname, dset, pr)
        ol = openloop(model, dq, dt, C, hist, mname, dset, pr); sync_datasets()
        print(f"{mname}/{dset} PR{pr}: built + open-loop in {time.time()-t0:.0f}s | "
              f"f̂(≥2^b) train: " + ", ".join(f"{b}b {ol[0][b]:.1e}" for b in range(3, 8)), flush=True)
        for ph, p, b, nb, kw in todo:
            run(ph, model, dt, C, cnt, mname, dset, pr, nb, b, ol=ol, **kw)
        del model; torch.cuda.empty_cache()


R18, C10 = "resnet18", "cifar10"
if RUN_REF:
    for mname, dset in ((R18, C10), ("vgg8", C10), (R18, "cifar100")):
        campaign(mname, dset, [("REF", 128, 8, N_FULL, {})])

if RUN_A:
    A = [("A", pr, 4, N_MID, {}) for pr in (16, 24, 32)]
    A += [("A", pr, 5, N_MID, {}) for pr in (32, 48, 96)]
    A += [("A", pr, 5, N_FULL, {}) for pr in (64, 72, 80, 88)]
    A += [("A", pr, 6, N_MID, {}) for pr in (80, 88)]
    A += [("A", pr, 6, N_FULL, {}) for pr in (96, 112, 128)]
    A += [("A", 128, 7, N_MID, {})]
    campaign(R18, C10, A)

if RUN_C:
    Cj = [("C", 128, b, N_QUICK, dict(s=s, rounding=r))
          for (b, s) in ((7, 1), (6, 1), (5, 1), (5, 2)) for r in ("halfup", "halfeven", "stochastic")]
    campaign(R18, C10, Cj)

if RUN_D:
    Dj = [("D", pr, b, N_QUICK, dict(noise=sg, noise_mode=nm))
          for (pr, b) in ((128, 6), (64, 5)) for nm in ("clamp0", "symmetric") for sg in (0.11, 0.14, 0.19, 0.25)]
    campaign(R18, C10, Dj)

if RUN_B:
    for mname, dset in (("vgg8", C10), (R18, "cifar100")):
        Bj = [("B", pr, 5, N_FULL, {}) for pr in (64, 72, 80, 88)]
        Bj += [("B", 96, 5, N_MID, {}), ("B", 128, 6, N_FULL, {})]
        campaign(mname, dset, Bj)

print("\nDONE. Send me S6_runs.csv, S6_openloop.csv and S6_hist.csv (Drive: NeuroSim15v_paper3/full_rerun_v1/).")

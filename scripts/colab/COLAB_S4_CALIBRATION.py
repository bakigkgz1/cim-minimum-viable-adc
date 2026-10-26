# === PAPER III — S4: FULL-SCALE (LSB) CALIBRATION OF THE ADC  (one Colab cell, T4 GPU) ===
# Idea: instead of adding converter bits, move the converter's full scale. A b-bit ADC with an
# LSB of 2^s cell units covers partial sums up to 2^(b+s), trading LSB size for range.
#   s = 0 : LSB = 1 cell (the paper's default; 5 bits clips at PR = 128)
#   s = 1 : LSB = 2 cells, s = 2 : LSB = 4 cells, ...
# Per-layer calibration picks, for every CIM layer, the smallest s whose clipped fraction stays
# below TAU on 256 training images measured WITHOUT clipping (so the profile is not that of a
# collapsed network). Evaluated at PR = 128 on 1024 test images, then the best plans on the full test.
#
# Approximate run time on a T4 (after setup, which is 0–8 min):
#   A  layer statistics (ResNet18, VGG8, ResNet18/C100)        ~ 5 min (+ CIFAR download if not cached on Drive)
#   B  12 plans x 1024 images, ResNet18 / CIFAR-10             ~ 20 min
#   C  full 10 000-image test of 3 plans (ResNet18 / C10)      ~ 35 min
#   D  per-layer plan, full test: VGG8 / C10 and ResNet18 / C100 ~ 25 min
#   Total ~ 1.5 h. Resume-safe: finished rows are skipped.
# Outputs: Drive/NeuroSim15v_paper3/full_rerun_v1/S4_calibration.csv, S4_layer_scale.csv
# altered_frac = share of conversions whose code is clamped (code > 2^b after LSB scaling).

TAU = 2e-5          # clipped-fraction budget per layer (the paper's accuracy-safe limit)
PR  = 128
RUN_C, RUN_D = True, True

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
S4  = OUT / "S4_calibration.csv"
S4L = OUT / "S4_layer_scale.csv"

def build(model_name, dataset, adc, seed=1234, tag="s4"):
    """Build the quantized model once; ADC bits and LSB scale are then set per layer at run time."""
    sys.argv = ["inference.py", "--dataset", dataset, "--model", model_name, "--hardware", "0", "--ppa", "0",
                "--num_batches", "16", "--batch_size", "64", "--calib_batch_size", "128", "--seed", str(seed),
                "--parallel_read", str(PR), "--adc_precision", str(adc), "--sub_array", "[128,128]",
                "--data_path", "./datasets/", "--model_path", "./models/", "--test_name", tag]
    args = parse_args()
    torch.cuda.set_device(args.gpu); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    args.data_path = args.data_path + args.dataset + "/"
    args.logdir = make_path.makepath(args, ["log_interval", "test_interval", "logdir", "epochs", "gpu",
                                            "ngpu", "debug", "data_path", "model_path"])
    misc.ensure_dir(args.logdir); misc.logger.init(args.logdir, "s4"); args.logger = misc.logger.info
    getter = get_cifar10 if dataset == "cifar10" else dataset_mod.get_cifar100
    dq = getter(args.calib_batch_size, args.data_path, train=True, val=False)
    dt = getter(args.batch_size, args.data_path, train=False, val=True)
    hook.make_records(args); args.hook = True; args.write_network = True
    model = quantize_model(args, nn.CrossEntropyLoss(), dq, dt).cuda().eval()

    cfg, hist, cnt = {}, {}, {"nelem": 0, "nclip": 0}
    mode = {"stats": False}
    for name, m in model.named_modules():
        if isinstance(m, macro.CIM) and hasattr(m, "_adc_quant"):
            cfg[name] = [adc, 0]                      # [bits, s]  (LSB = 2^s cells)
            orig = m._adc_quant
            def wrapped(self, x, _orig=orig, _n=name):
                if self._adc_quantizer is None or self._adc_quantizer._disabled:
                    return _orig(x)
                b, s = cfg[_n]
                if mode["stats"]:                     # record the unclipped partial-sum histogram
                    v = x.detach().round().clamp(0, PR).long().flatten()   # partition sums lie in [0, PR]
                    h = torch.bincount(v, minlength=PR + 1).cpu()
                    hist[_n] = hist[_n] + h if _n in hist else h
                    self._adc_quantizer.num_bits = 12
                    return _orig(x)
                self._adc_quantizer.num_bits = b
                L = float(2 ** s)
                q = torch.floor(x / L + 0.5)          # b-bit converter with an LSB of L cell units
                cnt["nelem"] += q.numel(); cnt["nclip"] += int((q > 2 ** b).sum().item())
                return _orig(q) * L                    # _orig clamps the code to [0, 2^b]
            m._adc_quant = types.MethodType(wrapped, m)
    return model, dq, dt, cfg, hist, cnt, mode

def evaluate(model, loader, nb):
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if nb >= 0 and i >= nb: break
            out = model(x.cuda()); correct += (out.argmax(1) == y.cuda()).sum().item(); total += y.numel()
    return round(100.0 * correct / max(total, 1), 4), total

def collect_stats(model, dq, mode, nb=2):
    mode["stats"] = True
    with torch.no_grad():
        for i, (x, _) in enumerate(dq):
            if i >= nb: break
            model(x.cuda())
    mode["stats"] = False

def clip_frac(h, b, s):
    v = torch.arange(h.numel(), dtype=torch.float64)
    code = torch.floor(v / 2 ** s + 0.5)
    return float(h[code > 2 ** b].sum()) / max(float(h.sum()), 1.0)

def per_layer_plan(hist, b, tau=TAU, smax=3):
    plan = {}
    for n, h in hist.items():
        plan[n] = next((s for s in range(smax + 1) if clip_frac(h, b, s) <= tau), smax)
    return plan

def conv_share(hist, plan):
    tot = sum(float(h.sum()) for h in hist.values())
    used = {n: plan.get(n, 0) for n in hist}
    return {s: round(sum(float(hist[n].sum()) for n in hist if used[n] == s) / tot, 4) for s in sorted(set(used.values()))}

def run_plan(model, dt, cfg, cnt, plan_bits, plan_s, nb):
    for n in cfg: cfg[n] = [plan_bits.get(n, max(plan_bits.values())), plan_s.get(n, 0)]
    cnt["nelem"] = cnt["nclip"] = 0
    acc, n = evaluate(model, dt, nb)
    return acc, n, (cnt["nclip"] / cnt["nelem"] if cnt["nelem"] else float("nan"))

def record(model_name, dataset, plan, b, sdict, hist, acc, n, clip, t0):
    shares = conv_share(hist, sdict)
    row = dict(ts=utc(), model=model_name, dataset=dataset, pr=PR, plan=plan, adc_bits=b,
               s_min=min(sdict.get(n, 0) for n in hist), s_max=max(sdict.get(n, 0) for n in hist),
               share_s0=shares.get(0, 0.0), share_s1=shares.get(1, 0.0), share_s2=shares.get(2, 0.0), share_s3=shares.get(3, 0.0),
               n_images=n, acc=acc, altered_frac=f"{clip:.8f}", seconds=round(time.time() - t0, 1))
    append_row(S4, row); print("S4", row, flush=True)

def save_layer_plan(model_name, dataset, b, plan, hist):
    for n, s in plan.items():
        if done(S4L, True, model=model_name, dataset=dataset, adc_bits=b, layer=n): continue
        append_row(S4L, dict(model=model_name, dataset=dataset, pr=PR, adc_bits=b, layer=n, s=s, lsb_cells=2 ** s,
                             conversions=int(hist[n].sum()), clip_s0=f"{clip_frac(hist[n], b, 0):.3e}",
                             clip_chosen=f"{clip_frac(hist[n], b, s):.3e}",
                             vmax=int((hist[n] > 0).nonzero().max()) if hist[n].sum() > 0 else 0))

# ================================================================ ResNet18 / CIFAR-10
t0 = time.time()
model, dq, dt, cfg, hist, cnt, mode = build("resnet18", "cifar10", 5, tag="s4_r18c10")
collect_stats(model, dq, mode); sync_datasets()
print(f"A: layer statistics for {len(hist)} converter-path layers ({time.time()-t0:.0f}s)", flush=True)
names = list(hist)                                         # the 3-channel stem never reaches a converter
plans = []
for b in (5, 4):
    for s in range(0, 4 if b == 4 else 3):
        plans.append((f"uniform_b{b}_s{s}", b, {n: s for n in names}))
    p = per_layer_plan(hist, b); save_layer_plan("resnet18", "cifar10", b, p, hist)
    plans.append((f"perlayer_b{b}", b, p))
plans.append(("uniform_b6_s0", 6, {n: 0 for n in names}))

for plan, b, sd in plans:                                  # B: 1024 images
    if done(S4, True, model="resnet18", dataset="cifar10", plan=plan, n_images=1024): print("skip", plan); continue
    t0 = time.time(); acc, n, clip = run_plan(model, dt, cfg, cnt, {k: b for k in names}, sd, 16)
    record("resnet18", "cifar10", plan, b, sd, hist, acc, n, clip, t0)

if RUN_C:                                                  # C: full test on the per-layer plans + best uniform 5-bit
    rows = [r for r in load_rows(S4) if r["model"] == "resnet18" and r["dataset"] == "cifar10" and r["n_images"] == "1024"]
    best_u5 = max([r for r in rows if r["plan"].startswith("uniform_b5_s") and r["plan"] != "uniform_b5_s0"],
                  key=lambda r: float(r["acc"]))["plan"]
    for plan in ("perlayer_b5", best_u5, "perlayer_b4"):
        if done(S4, True, model="resnet18", dataset="cifar10", plan=plan, n_images=10000): print("skip full", plan); continue
        b, sd = next((b, sd) for p, b, sd in plans if p == plan)
        t0 = time.time(); acc, n, clip = run_plan(model, dt, cfg, cnt, {k: b for k in names}, sd, -1)
        record("resnet18", "cifar10", plan, b, sd, hist, acc, n, clip, t0)
del model; torch.cuda.empty_cache()

# ================================================================ generality: VGG8 / C10 and ResNet18 / C100
if RUN_D:
    for mname, dset in (("vgg8", "cifar10"), ("resnet18", "cifar100")):
        need = [p for p in ("uniform_b5_s0", "perlayer_b5", "uniform_b6_s0")
                if not done(S4, True, model=mname, dataset=dset, plan=p, n_images=10000)]
        if not need: print("skip", mname, dset); continue
        t0 = time.time()
        model, dq, dt, cfg, hist, cnt, mode = build(mname, dset, 5, tag=f"s4_{mname}_{dset}")
        collect_stats(model, dq, mode); sync_datasets(); names = list(hist)
        p5 = per_layer_plan(hist, 5); save_layer_plan(mname, dset, 5, p5, hist)
        print(f"{mname}/{dset}: stats {time.time()-t0:.0f}s", flush=True)
        for plan in need:
            b, sd = {"uniform_b5_s0": (5, {n: 0 for n in names}), "perlayer_b5": (5, p5),
                     "uniform_b6_s0": (6, {n: 0 for n in names})}[plan]
            nb = 16 if plan != "perlayer_b5" else -1          # references on 1024 images, calibrated plan on the full test
            if plan != "perlayer_b5" and done(S4, True, model=mname, dataset=dset, plan=plan, n_images=1024): continue
            t0 = time.time(); acc, n, clip = run_plan(model, dt, cfg, cnt, {k: b for k in names}, sd, nb)
            record(mname, dset, plan, b, sd, hist, acc, n, clip, t0)
        del model; torch.cuda.empty_cache()

print("\nDONE. Send me the S4 lines of this output (or S4_calibration.csv and S4_layer_scale.csv).")

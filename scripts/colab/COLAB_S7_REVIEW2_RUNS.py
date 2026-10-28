# === PAPER III — S7: SECOND-REVIEW RUNS (one Colab cell, T4 GPU) ===
# Same converter as S6 (textbook b-bit: 2^b codes, clamp [0, 2^b - 1]). Answers the second review:
#   P  PR24/4-bit on the full test (the lossy point that bounds tau was on 2 048 images)
#   Q  PR128/5-bit for all three networks with the textbook converter (2 048 images)
#   L  per-layer feedback: open-loop vs closed-loop clipped fraction and mean partial sum of every
#      CIM layer (ResNet18/C10, PR 72 and 80, 5-bit, same 256 test images)
#   K  single-layer clipping: only one layer gets 5 bits (all others 8 bits, which cannot clip at PR 80);
#      records how the would-be-clipped share of every downstream layer changes
#   N  conductance read noise on the circuit-model path (hardware=1): does device noise shift the boundary?
#      ResNet18/C10, PR64/5, PR72/5, PR128/6, PR128/8 x read noise {0, 0.005, 0.01, 0.02, 0.03}, 1024 images
#   S  calibration-seed repeats (seeds 2345, 3456) of the boundary points and the 8-bit references,
#      5 120 images: ResNet18 PR64/5, PR72/5; VGG8 PR64/5, PR72/5; ResNet18/C100 PR64/5
#
# Approximate run time on a T4 (setup 0–8 min; resume-safe — rerun the cell after a disconnect):
#   P ~1.5 h | Q ~25 min | L ~15 min | K ~25 min | N ~1.5 h | S ~5 h      Total ~9–10 h
#   Order: P, Q, L, K, N, S. Switch phases off below to split the work across sessions.
# Outputs (Drive/NeuroSim15v_paper3/full_rerun_v1/):
#   S7_runs.csv, S7_layers.csv, S7_openloop.csv, S7_preds/*.npy

RUN_P, RUN_Q, RUN_L, RUN_K, RUN_N, RUN_S = True, True, True, True, True, True
N_FULL, N_MID, N_QUICK, N_SEED = -1, 32, 16, 80     # batches of 64: full test / 2048 / 1024 / 5120 images
SEEDS = (2345, 3456)
READ_NOISE = (0.0, 0.005, 0.01, 0.02, 0.03)

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
S7R = OUT / "S7_runs.csv"; S7L = OUT / "S7_layers.csv"; S7O = OUT / "S7_openloop.csv"
PRED = OUT / "S7_preds"; PRED.mkdir(exist_ok=True)
DONE_KEYS = {r["key"] for r in load_rows(S7R)}
L_DONE = {r["exp"] for r in load_rows(S7L)}
OL_DONE = {r["cfg"] for r in load_rows(S7O)}


def build(model_name, dataset, pr, seed=1234, hardware=0, adc=8):
    sys.argv = ["inference.py", "--dataset", dataset, "--model", model_name, "--hardware", str(hardware), "--ppa", "0",
                "--num_batches", "16", "--batch_size", "64", "--calib_batch_size", "128", "--seed", str(seed),
                "--parallel_read", str(pr), "--adc_precision", str(adc), "--sub_array", "[128,128]",
                "--data_path", "./datasets/", "--model_path", "./models/",
                "--test_name", f"s7_{model_name}_{dataset}_{pr}_{seed}_{hardware}"]
    args = parse_args()
    torch.cuda.set_device(args.gpu); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    args.data_path = args.data_path + args.dataset + "/"
    args.logdir = make_path.makepath(args, ["log_interval", "test_interval", "logdir", "epochs", "gpu",
                                            "ngpu", "debug", "data_path", "model_path"])
    misc.ensure_dir(args.logdir); misc.logger.init(args.logdir, "s7"); args.logger = misc.logger.info
    getter = get_cifar10 if dataset == "cifar10" else dataset_mod.get_cifar100
    dq = getter(args.calib_batch_size, args.data_path, train=True, val=False)
    dt = getter(args.batch_size, args.data_path, train=False, val=True)
    hook.make_records(args); args.hook = True; args.write_network = True
    model = quantize_model(args, nn.CrossEntropyLoss(), dq, dt).cuda().eval()

    C = dict(stats=False, bits=8, only=None, probe=5, read_noise=0.0)
    hist, cnt, lay = {}, {"nelem": 0, "nalt": 0}, {}
    names, mods = [], []
    for name, m in model.named_modules():
        if isinstance(m, macro.CIM) and hasattr(m, "_adc_quant"):
            names.append(name); mods.append(m)
            orig = m._adc_quant
            def wrapped(self, x, _orig=orig, _n=name):
                if self._adc_quantizer is None or self._adc_quantizer._disabled or self._adc_quantizer._if_calib:
                    return _orig(x)
                if C["stats"]:
                    v = x.detach().round().clamp(0, pr).long().flatten()
                    h = torch.bincount(v, minlength=pr + 1).cpu()
                    hist[_n] = hist[_n] + h if _n in hist else h
                    return x
                b = C["bits"] if (C["only"] is None or C["only"] == _n) else 8
                top = 2 ** b - 1
                xd = x.detach()
                rec = lay.setdefault(_n, [0, 0, 0.0, 0])           # n, n >= 2^probe, sum y, n altered
                rec[0] += xd.numel(); rec[1] += int((xd >= 2 ** C["probe"]).sum().item())
                rec[2] += float(xd.sum().item()); alt = int((xd > top).sum().item()); rec[3] += alt
                cnt["nelem"] += xd.numel(); cnt["nalt"] += alt
                return x.clamp(0, top)
            m._adc_quant = types.MethodType(wrapped, m)
    return model, dq, dt, C, hist, cnt, lay, names, mods


def set_hw(mods, bits, read_noise):
    for m in mods:
        m._cim_args.read_noise = float(read_noise)
        m._cim_args.adc_precision = int(bits)       # the conductance model clamps at 2^adc_precision first


def openloop_tail(model, dq, C, hist, tag):
    """Open-loop clipped fraction on 512 training images for this build (seed-specific quantization)."""
    if tag in OL_DONE: return
    C["stats"] = True; hist.clear()
    with torch.no_grad():
        for i, (x, _) in enumerate(dq):
            if i >= 4: break
            model(x.cuda())
    C["stats"] = False
    h = sum(hist.values()); v = torch.arange(h.numel()); hist.clear()
    row = dict(cfg=tag, conversions=int(h.sum()))
    for b in range(2, 9): row[f"ge_{2**b}"] = int(h[v >= 2 ** b].sum())
    append_row(S7O, row); OL_DONE.add(tag)
    print("open-loop", tag, ", ".join(f"{b}b {row[f'ge_{2**b}']/row['conversions']:.2e}" for b in range(3, 8)), flush=True)


def evaluate(model, dt, nb, seed=1234):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    correct = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dt):
            if nb >= 0 and i >= nb: break
            correct.append((model(x.cuda()).argmax(1) == y.cuda()).cpu().numpy())
    return np.concatenate(correct)


def run(phase, model, dt, C, cnt, mname, dset, pr, nb, bits, seed=1234, hw=0, rn=0.0, mods=None):
    key = f"{phase}|{mname}|{dset}|pr{pr}|b{bits}|seed{seed}|hw{hw}|rn{rn}|nb{nb}"
    if key in DONE_KEYS: print("skip", key); return
    C.update(bits=bits, only=None, probe=bits)
    if hw: set_hw(mods, bits, rn)
    cnt["nelem"] = cnt["nalt"] = 0; t0 = time.time()
    c = evaluate(model, dt, nb, seed)
    fname = re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".npy"; np.save(PRED / fname, c)
    row = dict(ts=utc(), key=key, phase=phase, model=mname, dataset=dset, pr=pr, bits=bits, seed=seed, hardware=hw,
               read_noise=rn, n_images=c.size, acc=round(100.0 * c.mean(), 4),
               altered_frac=f"{cnt['nalt'] / max(cnt['nelem'], 1):.3e}", seconds=round(time.time() - t0, 1), pred_file=fname)
    append_row(S7R, row); DONE_KEYS.add(key)
    print(f"S7 {phase} {mname}/{dset} PR{pr} {bits}b seed{seed} hw{hw} rn{rn}: acc {row['acc']} (n={c.size}) "
          f"altered {row['altered_frac']} [{row['seconds']}s]", flush=True)


def layer_record(exp, model, dt, C, hist, lay, names, bits, only=None, probe=5, nb=4):
    """Per-layer open-loop (exact sums) and closed-loop statistics on the same nb*64 test images."""
    if exp in L_DONE: print("skip", exp); return
    C["stats"] = True; hist.clear()
    with torch.no_grad():
        for i, (x, _) in enumerate(dt):
            if i >= nb: break
            model(x.cuda())
    C["stats"] = False; ol = {k: v.clone() for k, v in hist.items()}; hist.clear()
    C.update(bits=bits, only=only, probe=probe); lay.clear()
    with torch.no_grad():
        for i, (x, _) in enumerate(dt):
            if i >= nb: break
            model(x.cuda())
    for idx, n in enumerate(names):
        if n not in ol or n not in lay: continue
        h = ol[n]; v = torch.arange(h.numel()); tot = float(h.sum())
        r = lay[n]
        append_row(S7L, dict(exp=exp, layer=n, index=idx, bits_layer=(bits if (only is None or only == n) else 8),
                             conversions=r[0], open_ge=f"{float(h[v >= 2 ** probe].sum()) / tot:.4e}",
                             closed_ge=f"{r[1] / max(r[0], 1):.4e}", closed_altered=f"{r[3] / max(r[0], 1):.4e}",
                             open_mean=f"{float((v * h).sum()) / tot:.4f}", closed_mean=f"{r[2] / max(r[0], 1):.4f}"))
    C["only"] = None; L_DONE.add(exp)
    print("layers", exp, "written", flush=True)


R18, C10, C100 = "resnet18", "cifar10", "cifar100"
def free(model):
    del model; torch.cuda.empty_cache()

if RUN_P:
    k = f"P|{R18}|{C10}|pr24|b4|seed1234|hw0|rn0.0|nb{N_FULL}"
    if k not in DONE_KEYS:
        model, dq, dt, C, hist, cnt, lay, names, mods = build(R18, C10, 24)
        run("P", model, dt, C, cnt, R18, C10, 24, N_FULL, 4); free(model)

if RUN_Q:
    for mname, dset in ((R18, C10), ("vgg8", C10), (R18, C100)):
        k = f"Q|{mname}|{dset}|pr128|b5|seed1234|hw0|rn0.0|nb{N_MID}"
        if k in DONE_KEYS: print("skip", k); continue
        model, dq, dt, C, hist, cnt, lay, names, mods = build(mname, dset, 128)
        run("Q", model, dt, C, cnt, mname, dset, 128, N_MID, 5); free(model)

if RUN_L or RUN_K:
    for pr in (72, 80):
        exps = []
        if RUN_L: exps.append(f"L_pr{pr}_b5_all")
        KL = ("conv2_x.0.residual_function.0", "conv3_x.0.residual_function.0", "conv4_x.0.residual_function.0", "conv5_x.0.residual_function.0")
        if RUN_K and pr == 80: exps += [f"K_pr80_only_{n}" for n in KL]
        if all(e in L_DONE for e in exps): print("skip layers PR", pr); continue
        model, dq, dt, C, hist, cnt, lay, names, mods = build(R18, C10, pr)
        sync_datasets()
        if RUN_L: layer_record(f"L_pr{pr}_b5_all", model, dt, C, hist, lay, names, 5, probe=5)
        if RUN_K and pr == 80:
            for n in KL:
                if n in names:
                    layer_record(f"K_pr80_only_{n}", model, dt, C, hist, lay, names, 5, only=n, probe=5)
                else:
                    print("layer not found:", n, "| available:", names[:6], "...")
        free(model)

if RUN_N:
    for pr, bits in ((64, 5), (72, 5), (128, 6), (128, 8)):
        todo = [rn for rn in READ_NOISE if f"N|{R18}|{C10}|pr{pr}|b{bits}|seed1234|hw1|rn{rn}|nb{N_QUICK}" not in DONE_KEYS]
        if not todo: print("skip N", pr, bits); continue
        model, dq, dt, C, hist, cnt, lay, names, mods = build(R18, C10, pr, hardware=1, adc=bits)
        for rn in todo:
            run("N", model, dt, C, cnt, R18, C10, pr, N_QUICK, bits, hw=1, rn=rn, mods=mods)
        free(model)

if RUN_S:
    for seed in SEEDS:
        for mname, dset, pts in ((R18, C10, (64, 72)), ("vgg8", C10, (64, 72)), (R18, C100, (64,))):
            for pr, bits in [(128, 8)] + [(p, 5) for p in pts]:
                k = f"S|{mname}|{dset}|pr{pr}|b{bits}|seed{seed}|hw0|rn0.0|nb{N_SEED}"
                if k in DONE_KEYS: print("skip", k); continue
                model, dq, dt, C, hist, cnt, lay, names, mods = build(mname, dset, pr, seed=seed)
                openloop_tail(model, dq, C, hist, f"{mname}|{dset}|pr{pr}|seed{seed}")
                sync_datasets()
                run("S", model, dt, C, cnt, mname, dset, pr, N_SEED, bits, seed=seed); free(model)

print("\nDONE. Send me S7_runs.csv, S7_layers.csv, S7_openloop.csv and a zip of S7_preds (Drive: NeuroSim15v_paper3/full_rerun_v1/).")

# === PAPER III — S5: PARTIAL-SUM HISTOGRAMS FOR THE ANALYTIC RESOLUTION RULE (one Colab cell, T4 GPU) ===
# Records, for every converter-path layer, the full histogram of partial sums (no clipping, 256 training
# images) at several read widths. From these histograms the minimum ADC resolution and the clipped
# fraction can be predicted in closed form and checked against the measured accuracy cliff.
# No accuracy runs: one model build + two forward batches per configuration.
#
# Approximate run time on a T4: ~1.5 min per configuration x 16 configurations = ~25 min
# (+ setup 0–8 min; CIFAR files are taken from the Drive cache made by S4).
# Output: Drive/NeuroSim15v_paper3/full_rerun_v1/S5_partial_sum_hist.csv

CONFIGS = [("resnet18", "cifar10", pr) for pr in (16, 32, 48, 64, 72, 80, 88, 96, 112, 128)] + \
          [("vgg8", "cifar10", pr) for pr in (64, 96, 128)] + \
          [("resnet18", "cifar100", pr) for pr in (64, 96, 128)]
PR = 128   # updated per configuration; the histogram wrapper reads it at run time

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
S5 = OUT / "S5_partial_sum_hist.csv"

def build(model_name, dataset, adc, pr, seed=1234, tag="s5"):
    """Build the quantized model once; ADC bits and LSB scale are then set per layer at run time."""
    sys.argv = ["inference.py", "--dataset", dataset, "--model", model_name, "--hardware", "0", "--ppa", "0",
                "--num_batches", "16", "--batch_size", "64", "--calib_batch_size", "128", "--seed", str(seed),
                "--parallel_read", str(pr), "--adc_precision", str(adc), "--sub_array", "[128,128]",
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

def collect_stats(model, dq, mode, nb=2):
    mode["stats"] = True
    with torch.no_grad():
        for i, (x, _) in enumerate(dq):
            if i >= nb: break
            model(x.cuda())
    mode["stats"] = False

for mname, dset, pr in CONFIGS:
    if done(S5, True, model=mname, dataset=dset, pr=pr): print("skip", mname, dset, pr); continue
    PR = pr; t0 = time.time()
    model, dq, dt, cfg, hist, cnt, mode = build(mname, dset, 12, pr, tag=f"s5_{mname}_{dset}_{pr}")
    collect_stats(model, dq, mode); sync_datasets()
    for n, h in hist.items():
        hh = h[:pr + 1].tolist()
        append_row(S5, dict(model=mname, dataset=dset, pr=pr, layer=n, conversions=int(sum(hh)),
                            hist=";".join(str(int(v)) for v in hh)))
    print(f"S5 {mname}/{dset} PR={pr}: {len(hist)} layers, {time.time()-t0:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()

print("\nDONE. Send me S5_partial_sum_hist.csv (Drive: NeuroSim15v_paper3/full_rerun_v1/).")

# === PAPER III — STRENGTHENING RUNS (S1 ENOB, S2 measured SAR, S3 mixed precision) ===
# Paste into ONE Colab cell (T4 GPU) or upload to Drive and %run it. Resume-safe:
# rows already in the output CSVs are skipped, so you can re-run after a disconnect.
#
# Approximate run time on a T4 (after setup):
#   setup (clone/pip/make, first time only)     ~5–8 min
#   S1  ENOB via ADC output noise, 12 runs x 1024 img   ~15–25 min
#   S2  matched-LO SAR hardware, 5 corners              ~25–35 min (≈4 min/run + rebuild)
#   S3  per-layer mixed precision, 8 runs x 1024 img    ~10–15 min
#       + full 10 000-image test of the 2 best maps     ~25 min
#   Total ≈ 1.5–2 h.  Flip the RUN_* flags to run only part of it.
#
# Outputs: Drive/NeuroSim15v_paper3/full_rerun_v1/
#   S1_enob_noise.csv   S2_sar_hw.csv   S3_mixed_precision.csv   logs/hw_SAR_*.log

RUN_S1, RUN_S2, RUN_S3, RUN_S3_FULL = True, True, True, True

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

# ---------------------------------------------------------------- accuracy runner
def run_eval(pr, adc, num_batches, hardware=0, output_noise=0.0, bit_map=None, seed=1234, tag="s"):
    sys.argv = ["inference.py", "--dataset", "cifar10", "--model", "resnet18",
                "--hardware", str(hardware), "--ppa", "0",
                "--num_batches", str(num_batches), "--batch_size", "64", "--calib_batch_size", "128",
                "--seed", str(seed), "--parallel_read", str(pr), "--adc_precision", str(adc),
                "--sub_array", "[128,128]", "--output_noise", str(output_noise),
                "--data_path", "./datasets/", "--model_path", "./models/", "--test_name", f"strengthen_{tag}"]
    args = parse_args()
    torch.cuda.set_device(args.gpu); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    args.data_path = args.data_path + args.dataset + "/"
    args.logdir = make_path.makepath(args, ["log_interval", "test_interval", "logdir", "epochs", "gpu",
                                            "ngpu", "debug", "data_path", "model_path"])
    misc.ensure_dir(args.logdir); misc.logger.init(args.logdir, "strengthen"); args.logger = misc.logger.info
    dq = get_cifar10(args.calib_batch_size, args.data_path, train=True, val=False)
    dt = get_cifar10(args.batch_size, args.data_path, train=False, val=True)
    hook.make_records(args); args.hook = True; args.write_network = True
    model = quantize_model(args, nn.CrossEntropyLoss(), dq, dt).cuda().eval()

    # per-layer converter resolution (S3); the hook also counts clipped values per layer
    stats = {"nelem": 0, "nclip": 0}
    names_set = []
    for name, m in model.named_modules():
        if isinstance(m, macro.CIM) and hasattr(m, "_adc_quant"):
            if bit_map is not None and name in bit_map:
                m._adc_quantizer.num_bits = int(bit_map[name]); names_set.append(name)
            orig = m._adc_quant
            def wrapped(self, x, _orig=orig):
                if self._adc_quantizer is not None and not self._adc_quantizer._disabled:
                    lim = float(2 ** int(self._adc_quantizer.num_bits)); xd = x.detach()
                    stats["nelem"] += xd.numel(); stats["nclip"] += int((xd >= lim).sum().item())
                return _orig(x)
            m._adc_quant = types.MethodType(wrapped, m)
    if bit_map is not None:
        missing = set(bit_map) - set(names_set)
        assert not missing, f"layer names not found: {sorted(missing)[:5]}"

    nb = len(dt) if num_batches < 0 else num_batches
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dt):
            if i >= nb: break
            out = model(x.cuda()); correct += (out.argmax(1) == y.cuda()).sum().item(); total += y.numel()
    return round(100.0 * correct / max(total, 1), 4), total, (stats["nclip"] / stats["nelem"] if stats["nelem"] else float("nan"))

# ================================================================ S1: ENOB via output noise
# NeuroSim's --output_noise adds Gaussian noise (in LSB) to the integer ADC code, then rounds and clamps.
# An ENOB deficit of D bits at a one-cell LSB corresponds to sigma = sqrt((4**D - 1)/12) LSB.
S1 = OUT / "S1_enob_noise.csv"
if RUN_S1:
    deficits = [0.0, 0.25, 0.5, 1.0]
    corners = [(64, 5), (128, 6), (128, 8)]
    for pr, adc in corners:
        for D in deficits:
            sigma = round(math.sqrt((4 ** D - 1) / 12), 4)
            if done(S1, True, pr=pr, adc=adc, enob_deficit=D): print("skip S1", pr, adc, D); continue
            t0 = time.time()
            hw = 1 if sigma > 0 else 0          # sigma=0 reproduces the integer path
            acc, n, clip = run_eval(pr, adc, 16, hardware=hw, output_noise=sigma, tag=f"S1_pr{pr}_b{adc}_d{D}")
            row = dict(ts=utc(), pr=pr, adc=adc, enob_deficit=D, sigma_lsb=sigma, hardware=hw,
                       n_images=n, acc=acc, seconds=round(time.time() - t0, 1))
            append_row(S1, row); print("S1", row, flush=True)

# ================================================================ S2: measured SAR converter (matched levels)
S2 = OUT / "S2_sar_hw.csv"
PARAM = ROOT / "NeuroSIM" / "Param.cpp"
BASE = DRIVE / "blok2d_Param_BASE.cpp"
if not BASE.exists():                       # fall back to the stock file of the pinned commit
    shutil.copy2(PARAM, BASE); print("BASE Param.cpp saved to Drive from stock")

def set_param(adc, sar):
    text = BASE.read_text()
    text, n1 = re.subn(r"levelOutput\s*=\s*\d+\s*;", f"levelOutput = {2**adc};", text, count=1)
    text, n2 = re.subn(r"SARADC\s*=\s*(true|false)\s*;", f"SARADC = {'true' if sar else 'false'};", text, count=1)
    assert n1 == 1 and n2 == 1, "Param.cpp pattern not found"
    PARAM.write_text(text)
    (LOG / f"Param_{'SAR' if sar else 'MLSA'}_ADC{adc}.cpp").write_text(text)
    assert os.system(f"cd {ROOT/'NeuroSIM'} && make -j2 -s") == 0, "NeuroSIM build failed"
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def last(rx, t):
    m = list(re.finditer(rx, t)); return m[-1].group(1) if m else ""

if RUN_S2:
    for pr, adc in [(64, 5), (128, 6), (128, 7), (128, 8), (64, 8)]:
        tag = f"SAR_PR{pr}_ADC{adc}"
        if done(S2, True, tag=tag): print("skip S2", tag); continue
        sha = set_param(adc, sar=True); t0 = time.time()
        logf = LOG / f"hw_{tag}.log"
        cmd = [sys.executable, "-u", str(ROOT / "inference.py"), "--dataset", "cifar10", "--model", "resnet18",
               "--hardware", "1", "--ppa", "1", "--num_batches", "1", "--batch_size", "64",
               "--parallel_read", str(pr), "--adc_precision", str(adc), "--sub_array", "[128,128]",
               "--data_path", "./datasets/", "--model_path", "./models/", "--test_name", f"hw_{tag}"]
        with logf.open("w") as f:
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT).returncode
        t = logf.read_text(errors="ignore"); assert rc == 0, t[-2000:]
        row = dict(ts=utc(), tag=tag, arch="SAR", pr=pr, adc=adc, levelOutput=2**adc, param_sha=sha,
                   tops_w=last(r"Energy Efficiency TOPS/W \(Pipelined Process\):\s*([\d.eE+-]+)", t),
                   fps=last(r"Throughput FPS \(Pipelined Process\):\s*([\d.eE+-]+)", t),
                   area=last(r"ChipArea\s*:\s*([\d.eE+-]+)", t),
                   adc_area=last(r"Total ADC \(or S/As and precharger for SRAM\) Area on chip\s*:\s*([\d.eE+-]+)", t),
                   adc_E_pJ=last(r"ADC \(or S/As and precharger for SRAM\) readDynamicEnergy is\s*:\s*([\d.eE+-]+)", t),
                   adc_latency_ns=last(r"ADC \(or S/As and precharger for SRAM\) readLatency is\s*:\s*([\d.eE+-]+)", t),
                   acc_1batch=last(r"Test Accuracy of the model on\s+\d+\s+test images:\s*([0-9.]+)", t),
                   seconds=round(time.time() - t0, 1))
        append_row(S2, row); print("S2", row, flush=True)
    set_param(7, sar=False)                 # restore stock-equivalent build (MLSA, 128 levels)
    print("Param.cpp restored to MLSA / levelOutput=128")

# ================================================================ S3: per-layer mixed precision at PR=128
# Rank layers from the measured PR128/5-bit dump (P2b): by clipped fraction c_l and by share s_l.
S3 = OUT / "S3_mixed_precision.csv"
if RUN_S3:
    rows = [r for r in load_rows(OUT / "P2b_clip_by_layer.csv")
            if r["model"] == "resnet18" and r["pr"] == "128" and r["adc"] == "5"]
    assert len(rows) == 20, f"expected 20 ResNet18 layers in P2b, got {len(rows)}"
    nelem = {r["layer"]: float(r["nelem"]) for r in rows}
    by_c = [r["layer"] for r in sorted(rows, key=lambda r: -float(r["clip_frac"]))]
    by_s = [r["layer"] for r in sorted(rows, key=lambda r: -float(r["nclip"]))]
    allL = list(nelem)

    def energy_rel(bmap):                    # SAR-model converter energy relative to uniform 6-bit
        return sum(nelem[l] * 2 ** bmap.get(l, 5) for l in allL) / sum(nelem[l] * 2 ** 6 for l in allL)

    plans = {"uniform5": {}}
    for k in (4, 8, 12, 16):
        plans[f"share_top{k}"] = {l: 6 for l in by_s[:k]}
        plans[f"rate_top{k}"]  = {l: 6 for l in by_c[:k]}
    plans["uniform6"] = {l: 6 for l in allL}

    for name, bm in plans.items():
        if done(S3, True, plan=name, n_images=1024): print("skip S3", name); continue
        full = {l: bm.get(l, 5) for l in allL}
        t0 = time.time()
        acc, n, clip = run_eval(128, 5, 16, hardware=0, bit_map=full, tag=f"S3_{name}")
        row = dict(ts=utc(), plan=name, layers_at_6bit=sum(1 for v in full.values() if v == 6),
                   conv_share_6bit=round(sum(nelem[l] for l in allL if full[l] == 6) / sum(nelem.values()), 4),
                   E_rel_uniform6_SAR=round(energy_rel(full), 4), n_images=n, acc=acc,
                   clip_frac=f"{clip:.8f}", layers_6bit=";".join(l for l in allL if full[l] == 6),
                   seconds=round(time.time() - t0, 1))
        append_row(S3, row); print("S3", row, flush=True)

    if RUN_S3_FULL:                          # full test on the two cheapest plans that are accuracy-safe at N=1024
        res = [r for r in load_rows(S3) if r["n_images"] == "1024" and r["plan"] not in ("uniform5", "uniform6")]
        safe = sorted([r for r in res if float(r["acc"]) >= 88.0], key=lambda r: float(r["E_rel_uniform6_SAR"]))[:2]
        for r in safe:
            if done(S3, True, plan=r["plan"], n_images=10000): continue
            full = {l: (6 if l in r["layers_6bit"].split(";") else 5) for l in allL}
            t0 = time.time()
            acc, n, clip = run_eval(128, 5, -1, hardware=0, bit_map=full, tag=f"S3full_{r['plan']}")
            row = dict(ts=utc(), plan=r["plan"], layers_at_6bit=r["layers_at_6bit"], conv_share_6bit=r["conv_share_6bit"],
                       E_rel_uniform6_SAR=r["E_rel_uniform6_SAR"], n_images=n, acc=acc, clip_frac=f"{clip:.8f}",
                       layers_6bit=r["layers_6bit"], seconds=round(time.time() - t0, 1))
            append_row(S3, row); print("S3 full", row, flush=True)

print("\nDONE. Send me these three CSVs:", S1.name, S2.name, S3.name)

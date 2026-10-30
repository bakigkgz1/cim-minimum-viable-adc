# === PAPER III — S12: REALISTIC CONVERTER ERRORS AND A DIFFERENTIAL BASELINE (one Colab cell, T4 GPU) ===
# Answers the two remaining reviewer points.
#   E  converter non-idealities on the functional path: every converter gets a static per-column
#      offset, static code-threshold (INL/DNL) errors and thermal noise, each with standard deviation
#      s (LSB); s = 0.05 / 0.075 / 0.10 / 0.15 at the headline points (ResNet18/CIFAR-10 PR128/5-bit exact
#      offset, PR128/6-bit analog unit column, 256-row PR256/6-bit exact offset), 2 048 images,
#      plus s = 0.05 on the full test set; and the analog unit column read by its own full-range
#      converter with the same errors (exempt mode 4)                                  ~3 h
#   D  differential baseline: weights mapped to positive and negative unsigned arrays (no unit
#      column; input shift removed digitally), each column converted by the b-bit converter;
#      ResNet18/CIFAR-10 at PR128 (4, 5, 6 bits) and 256-row PR256 (5, 6 bits), VGG8/CIFAR-10 at
#      PR128 (4, 5 bits), open-loop histograms, full test at the smallest safe-looking resolution  ~3 h
# Approximate total run time on a T4: ~6 h (resume-safe; restart the runtime first if another
# cell ran in this session). Outputs (Drive/NeuroSim15v_paper3/full_rerun_v1/): S12_runs.csv,
#   S12_openloop.csv, S12_hist.csv, S12_preds/*.npy

N_FULL, N_MID, N_QUICK, N_SEED = -1, 32, 16, 80

import zlib, csv, os, re, shutil, subprocess, sys, time, types, hashlib, math
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

# VGG8 on CIFAR-100 is not wired in the simulator's quantize.py; add it (100-way classifier).
QPY = ROOT / "quantize.py"; _q = QPY.read_text()
_qa = "        if args.model == 'resnet18':\n            model = resnet.resnet18(num_classes=100)\n"
if "vgg8_c100_patch" not in _q:
    assert _q.count(_qa) == 1, "quantize.py differs from the expected source; send me lines 100-112"
    QPY.write_text(_q.replace(_qa, _qa + "        elif args.model == 'vgg8':  # vgg8_c100_patch\n"
                              "            model = vgg.VGG(vgg.make_layers(vgg.cfg_list['vgg8']), num_classes=100)\n"))
import torch, torch.nn as nn
import pytorch_quantization.cim.modules.macro as macro
from inference import parse_args
import importlib, quantize as _quantize_mod
_quantize_mod = importlib.reload(_quantize_mod)   # pick up the VGG8/CIFAR-100 patch even if quantize was imported earlier
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
S8R = OUT / "S12_runs.csv"; S8O = OUT / "S12_openloop.csv"; S8H = OUT / "S12_hist.csv"
S8L = OUT / "S12_layers.csv"; S8W = OUT / "S12_hw.csv"
PRED = OUT / "S12_preds"; PRED.mkdir(exist_ok=True)
DONE_KEYS = {r["key"] for r in load_rows(S8R)}
L_DONE = {r["exp"] for r in load_rows(S8L)}
OL_DONE = {r["cfg"] for r in load_rows(S8O)}
HW_DONE = {r["tag"] for r in load_rows(S8W)}
MS10 = DRIVE / "models_s10"; MS10.mkdir(exist_ok=True)

TRAIN_PY = r"""
import sys, time, csv, argparse, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, "/content/NeuroSim15v")
from models import resnet, vgg
import torchvision.transforms as T, torchvision.datasets as DS
p = argparse.ArgumentParser()
p.add_argument("--arch"); p.add_argument("--dataset"); p.add_argument("--seed", type=int); p.add_argument("--epochs", type=int)
p.add_argument("--lr", type=float); p.add_argument("--min_acc", type=float); p.add_argument("--out"); p.add_argument("--log")
a = p.parse_args(); torch.manual_seed(a.seed); dev = "cuda"
nc = 10 if a.dataset == "cifar10" else 100; D = DS.CIFAR10 if nc == 10 else DS.CIFAR100
tr = T.Compose([T.Pad(4), T.RandomCrop(32), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
te = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
root = "/content/train_data"
dl = torch.utils.data.DataLoader(D(root, True, download=True, transform=tr), batch_size=128, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
dv = torch.utils.data.DataLoader(D(root, False, download=True, transform=te), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
m = (resnet.resnet18(num_classes=nc) if a.arch == "resnet18" else vgg.VGG(vgg.make_layers(vgg.cfg_list["vgg8"]), num_classes=nc)).to(dev)
opt = torch.optim.SGD(m.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs); scaler = torch.amp.GradScaler("cuda")
crit = nn.CrossEntropyLoss(); best = 0.0
for ep in range(1, a.epochs + 1):
    t0 = time.time(); m.train()
    for x, y in dl:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = crit(m(x), y)
        scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        scaler.step(opt); scaler.update()
    sch.step(); m.eval(); c = n = 0
    with torch.no_grad():
        for x, y in dv:
            c += (m(x.to(dev)).argmax(1).cpu() == y).sum().item(); n += y.numel()
    acc = 100.0 * c / n
    if acc > best: best = acc; torch.save(m.state_dict(), a.out)
    print(f"{a.arch}/{a.dataset} seed{a.seed} epoch {ep}/{a.epochs} test {acc:.2f}% best {best:.2f}% [{time.time()-t0:.0f}s]", flush=True)
    with open(a.log, "a", newline="") as f: csv.writer(f).writerow([a.arch, a.dataset, a.seed, ep, f"{acc:.2f}", f"{best:.2f}"])
    if acc >= a.min_acc: break
print("DONE", best, flush=True)
"""
Path("/content/train_s10.py").write_text(TRAIN_PY)

def train(arch, dset, seed, epochs, lr, min_acc, name):
    final = MS10 / f"{name}.pth"; flag = MS10 / f"{name}.done"
    if flag.exists(): print("skip training", name); return final
    tmp = Path(f"/content/{name}.pth")
    cmd = [sys.executable, "-u", "/content/train_s10.py", "--arch", arch, "--dataset", dset, "--seed", str(seed),
           "--epochs", str(epochs), "--lr", str(lr), "--min_acc", str(min_acc), "--out", str(tmp), "--log", str(OUT / "S10_train_log.csv")]
    t0 = time.time(); p = subprocess.run(cmd, cwd="/content")
    if p.returncode != 0: raise RuntimeError(f"training {name} failed")
    shutil.copy2(tmp, final); flag.write_text(f"{time.time()-t0:.0f}s"); print("trained", name, flush=True)
    return final

def ckdir(src, model_name, dset, tag):
    d = ROOT / f"models_{tag}"; d.mkdir(exist_ok=True)
    shutil.copy2(src, d / f"{model_name}_{dset}.pth"); return f"./models_{tag}/"

# ---------------------------------------------------------------- differential mapping (no unit column)
MODE = dict(diff=False)
def _stock_simulate_array():
    """Return NeuroSim's own CIM.simulate_array, whatever earlier cells have patched into the class.
    The code object is taken from macro.py itself, so a version patched by an earlier run of this
    (or an older) cell in the same session can never be picked up as the original."""
    cur = macro.CIM.__dict__["simulate_array"]
    if os.path.basename(cur.__code__.co_filename) == "macro.py":
        return cur
    src = open(macro.__file__).read()
    def find(code, name):
        for c in code.co_consts:
            if isinstance(c, types.CodeType):
                if c.co_name == name: return c
                r = find(c, name)
                if r is not None: return r
    cls = find(compile(src, macro.__file__, "exec"), "CIM")
    fn = types.FunctionType(find(cls, "simulate_array"), macro.__dict__, "simulate_array")
    print("S12: restored the stock simulate_array from macro.py (an earlier patch was active)")
    return fn
_S12_STOCK_SIM = _stock_simulate_array()
assert os.path.basename(_S12_STOCK_SIM.__code__.co_filename) == "macro.py"
def _sim(self, input2d, weight2d, _stock=_S12_STOCK_SIM):   # bound at definition: immune to later global renames
    if not MODE["diff"] or getattr(self._cim_args, "hook", False):   # first pass writes the layer records
        return _stock(self, input2d, weight2d)
    a = self._cim_args
    shift_in = -torch.min(input2d)                         # input shift removed digitally (zero after a ReLU)
    x = input2d + shift_in
    K = weight2d.shape[0]; nz = a.num_arrays_y * a.parallel_read - K; W = weight2d
    if nz > 0:
        x = torch.cat((x, torch.zeros((x.shape[0], nz), device=x.device)), dim=1)
        W = torch.cat((W, torch.zeros((nz, W.shape[1]), device=W.device)), dim=0)
    x = x.reshape(x.shape[0], a.num_arrays_y, -1).transpose(0, 1)
    Wp = W.clamp(min=0).reshape(a.num_arrays_y, -1, W.shape[1]).to(torch.int32)      # positive array
    Wn = (-W).clamp(min=0).reshape(a.num_arrays_y, -1, W.shape[1]).to(torch.int32)   # negative array
    xi = x.to(torch.int32); out = torch.zeros((x.shape[1], W.shape[1]), device=x.device)
    for i in range(a.cycles_per_input):
        xin = ((xi >> a.dac_precision * i) & (a.input_base - 1)).to(torch.float32)
        ps = torch.zeros_like(out)
        for j in range(a.cells_per_weight):
            wp = ((Wp >> a.bitcell * j) & (a.base - 1)).to(torch.float32)
            wn = ((Wn >> a.bitcell * j) & (a.base - 1)).to(torch.float32)
            qp = self._adc_quant(torch.matmul(xin, wp)); qn = self._adc_quant(torch.matmul(xin, wn))
            ps += (qp.sum(dim=0) - qn.sum(dim=0)).to(out.device) * (a.base ** j)
        out += ps * (a.input_base ** i)
    return out - shift_in * weight2d.sum(dim=0, keepdim=True)
macro.CIM.simulate_array = _sim

def build(model_name, dataset, pr, sub=128, seed=1234, mpath="./models/"):
    sys.argv = ["inference.py", "--dataset", dataset, "--model", model_name, "--hardware", "0", "--ppa", "0",
                "--num_batches", "16", "--batch_size", "64", "--calib_batch_size", "128", "--seed", str(seed),
                "--parallel_read", str(pr), "--adc_precision", "8", "--sub_array", f"[{sub},{sub}]",
                "--data_path", "./datasets/", "--model_path", mpath,
                "--test_name", f"s10_{model_name}_{dataset}_{pr}_{sub}_{Path(mpath).name}"]
    args = parse_args()
    torch.cuda.set_device(args.gpu); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    args.data_path = args.data_path + args.dataset + "/"
    args.logdir = make_path.makepath(args, ["log_interval", "test_interval", "logdir", "epochs", "gpu",
                                            "ngpu", "debug", "data_path", "model_path"])
    misc.ensure_dir(args.logdir); misc.logger.init(args.logdir, "s8"); args.logger = misc.logger.info
    getter = get_cifar10 if dataset == "cifar10" else dataset_mod.get_cifar100
    dq = getter(args.calib_batch_size, args.data_path, train=True, val=False)
    dt = getter(args.batch_size, args.data_path, train=False, val=True)
    hook.make_records(args); args.hook = True; args.write_network = True
    model = quantize_model(args, nn.CrossEntropyLoss(), dq, dt).cuda().eval()

    # C: converter settings. exempt: 0 = clip everything, 1 = dummy row and column converted without clipping,
    # 2 = dummy row only, 3 = dummy column only (the dummy row/column remove the input/weight offset)
    C = dict(stats=False, bits=8, only=None, probe=5, exempt=0, noise=0.0, err=0.0)
    hist, cnt, lay = {}, {"nelem": 0, "nalt": 0, "ndum": 0, "ndum_alt": 0, "nflip": 0}, {}
    hsig, hdum = {}, {}
    names = []; ERRC = {}
    def static_err(_n, b, A, M, dev, wide=False):
        # static per-column offset (A, 1, M) and sorted code thresholds; seeded per layer and resolution
        key = (_n, b, A, M, wide, C["err"])
        if key not in ERRC:
            g = torch.Generator(device="cpu"); g.manual_seed(zlib.crc32(_n.encode()) % 100003 * 97 + b * 13 + (7 if wide else 0))
            off = torch.randn((A, 1, M), generator=g) * C["err"]
            ncode = (pr + 1) if wide else 2 ** b
            thr = torch.arange(1, ncode, dtype=torch.float32) - 0.5 + torch.randn(ncode - 1, generator=g) * C["err"]
            ERRC[key] = (off.to(dev), torch.sort(thr).values.to(dev))
        return ERRC[key]
    def convert(y, b, _n, wide=False):
        # textbook b-bit converter, optionally with offset, INL/DNL and thermal noise (s = C["err"] LSB)
        if C["err"] <= 0:
            return y.clamp(0, (pr if wide else 2 ** b - 1))
        A, M = y.shape[0], y.shape[-1]
        off, thr = static_err(_n, b, A, M, y.device, wide)
        v = y + (off if y.dim() == 3 else 0.0) + C["err"] * torch.randn_like(y)
        return torch.bucketize(v.contiguous(), thr).to(y.dtype)
    for name, m in model.named_modules():
        if isinstance(m, macro.CIM) and hasattr(m, "_adc_quant"):
            names.append(name)
            orig = m._adc_quant
            def wrapped(self, x, _orig=orig, _n=name):
                if self._adc_quantizer is None or self._adc_quantizer._disabled or self._adc_quantizer._if_calib:
                    return _orig(x)
                split = x.dim() == 3 and not MODE["diff"]                # offset-encoded tensor with dummy row/column
                if C["stats"]:
                    xr = x.detach().round().clamp(0, pr).long()
                    h = torch.bincount(xr.flatten(), minlength=pr + 1).cpu()
                    hist[_n] = hist[_n] + h if _n in hist else h
                    hs = torch.bincount(xr[:, :-1, :-1].flatten(), minlength=pr + 1).cpu() if split else h
                    hsig[_n] = hsig[_n] + hs if _n in hsig else hs
                    hd = h - hs; hdum[_n] = hdum[_n] + hd if _n in hdum else hd
                    return x
                b = C["bits"] if (C["only"] is None or C["only"] == _n) else 8
                top = 2 ** b - 1
                y = x.detach(); y0 = y
                rec = lay.setdefault(_n, [0, 0, 0.0, 0])
                rec[0] += y.numel(); rec[1] += int((y >= 2 ** C["probe"]).sum().item())
                rec[2] += float(y.sum().item()); alt = int((y > top).sum().item()); rec[3] += alt
                cnt["nelem"] += y.numel(); cnt["nalt"] += alt
                q = convert(y, b, _n)
                cnt["nflip"] += int((q != y.clamp(0, top)).sum().item())  # codes changed by converter errors
                if split:
                    dr, dc = y[:, -1, :], y[:, :-1, -1]
                    cnt["ndum"] += dr.numel() + dc.numel()
                    cnt["ndum_alt"] += int((dr > top).sum().item() + (dc > top).sum().item())
                    if C["exempt"]:
                        q = q.clone()
                        if C["exempt"] == 4:                              # own full-range converter, same errors
                            qw = convert(y0, b, _n, wide=True)
                            q[:, -1, :] = qw[:, -1, :]; q[:, :, -1] = qw[:, :, -1]
                        else:
                            if C["exempt"] in (1, 2): q[:, -1, :] = y0[:, -1, :]
                            if C["exempt"] in (1, 3): q[:, :, -1] = y0[:, :, -1]
                return q
            m._adc_quant = types.MethodType(wrapped, m)
    C["hsig"], C["hdum"] = hsig, hdum
    return model, dq, dt, C, hist, cnt, lay, names


def openloop(model, dq, C, hist, tag, pr, keep_layers=False):
    if tag in OL_DONE and not keep_layers: return
    C["stats"] = True; hist.clear(); C["hsig"].clear(); C["hdum"].clear()
    with torch.no_grad():
        for i, (x, _) in enumerate(dq):
            if i >= 4: break
            model(x.cuda())
    C["stats"] = False
    if tag not in OL_DONE:
        for part, hd in (("all", hist), ("signal", C["hsig"]), ("offset", C["hdum"])):
            if not hd: continue
            h = sum(hd.values()); v = torch.arange(h.numel())
            row = dict(cfg=tag if part == "all" else f"{tag}|{part}", conversions=int(h.sum()))
            for b in range(2, 10): row[f"ge_{2**b}"] = int(h[v >= 2 ** b].sum())
            append_row(S8O, row)
            print("open-loop", row["cfg"], ", ".join(f"{b}b {row[f'ge_{2**b}']/max(row['conversions'],1):.2e}" for b in range(4, 10)), flush=True)
        OL_DONE.add(tag)
        for n, hh in hist.items():
            hl = hh[:pr + 1].tolist()
            append_row(S8H, dict(cfg=tag, layer=n, conversions=int(sum(hl)), hist=";".join(str(int(a)) for a in hl)))
    hist.clear()


def run(phase, model, dt, C, cnt, mname, dset, pr, nb, bits, sub=128, exempt=0, noise=0.0, ck="orig", err=0.0, diff=False):
    key = f"{phase}|{mname}|{dset}|{ck}|sub{sub}|pr{pr}|b{bits}|ex{int(exempt)}|e{err}|d{int(diff)}|nb{nb}"
    if key in DONE_KEYS: print("skip", key); return
    C.update(bits=bits, only=None, probe=bits, exempt=exempt, noise=noise, err=err); MODE["diff"] = diff
    for k in cnt: cnt[k] = 0
    torch.manual_seed(1234); torch.cuda.manual_seed_all(1234); t0 = time.time(); correct = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dt):
            if nb >= 0 and i >= nb: break
            correct.append((model(x.cuda()).argmax(1) == y.cuda()).cpu().numpy())
    c = np.concatenate(correct)
    fname = re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".npy"; np.save(PRED / fname, c)
    row = dict(ts=utc(), key=key, phase=phase, model=mname, dataset=dset, ckpt=ck, sub_array=sub, pr=pr, bits=bits,
               exempt_dummy=int(exempt), sigma_lsb=noise, adc_err_lsb=err, differential=int(diff), n_images=c.size, acc=round(100.0 * c.mean(), 4),
               altered_frac=f"{cnt['nalt'] / max(cnt['nelem'], 1):.3e}",
               dummy_altered_frac=f"{cnt['ndum_alt'] / max(cnt['ndum'], 1):.3e}",
               code_change_frac=f"{cnt['nflip'] / max(cnt['nelem'], 1):.3e}",
               seconds=round(time.time() - t0, 1), pred_file=fname)
    append_row(S8R, row); DONE_KEYS.add(key); C.update(exempt=0, noise=0.0, err=0.0); MODE["diff"] = False
    print(f"S12 {phase} {mname}/{dset} sub{sub} PR{pr} {bits}b exempt={int(exempt)} err={err} diff={int(diff)}: acc {row['acc']} "
          f"(n={c.size}) altered {row['altered_frac']} dummy-altered {row['dummy_altered_frac']} "
          f"code-change {row['code_change_frac']} [{row['seconds']}s]", flush=True)


def layer_record(exp, model, dt, C, hist, lay, names, bits, only=None, exempt=0, nb=4):
    if exp in L_DONE: print("skip", exp); return
    C["stats"] = True; hist.clear()
    with torch.no_grad():
        for i, (x, _) in enumerate(dt):
            if i >= nb: break
            model(x.cuda())
    C["stats"] = False; ol = {k: v.clone() for k, v in hist.items()}; hist.clear()
    C.update(bits=bits, only=only, probe=5, exempt=exempt); lay.clear()
    with torch.no_grad():
        for i, (x, _) in enumerate(dt):
            if i >= nb: break
            model(x.cuda())
    for idx, n in enumerate(names):
        if n not in ol or n not in lay: continue
        h = ol[n]; v = torch.arange(h.numel()); tot = float(h.sum()); r = lay[n]
        append_row(S8L, dict(exp=exp, layer=n, index=idx, bits_layer=(bits if (only is None or only == n) else 8),
                             conversions=r[0], open_ge=f"{float(h[v >= 32].sum()) / tot:.4e}",
                             closed_ge=f"{r[1] / max(r[0], 1):.4e}", open_mean=f"{float((v * h).sum()) / tot:.4f}",
                             closed_mean=f"{r[2] / max(r[0], 1):.4f}"))
    C.update(only=None, exempt=0); L_DONE.add(exp); print("layers", exp, "written", flush=True)


def free(model):
    del model; torch.cuda.empty_cache()

R18, C10, C100, V8 = "resnet18", "cifar10", "cifar100", "vgg8"

def point(phase, mname, dset, pr, sub, pts, ck="orig", mpath="./models/"):
    """pts: list of (bits, exempt, nb, noise)"""
    tag = f"{mname}|{dset}|{ck}|sub{sub}|pr{pr}"
    todo = [p for p in pts if f"{phase}|{mname}|{dset}|{ck}|sub{sub}|pr{pr}|b{p[0]}|ex{p[1]}|n{p[3]}|nb{p[2]}" not in DONE_KEYS]
    if not todo and tag in OL_DONE: print("skip", tag); return
    model, dq, dt, C, hist, cnt, lay, names = build(mname, dset, pr, sub=sub, mpath=mpath)
    openloop(model, dq, C, hist, tag, pr); sync_datasets()
    for b, ex, nb, sg in todo: run(phase, model, dt, C, cnt, mname, dset, pr, nb, b, sub=sub, exempt=ex, noise=sg, ck=ck)
    free(model)

def point12(phase, mname, dset, pr, sub, pts, ck="orig", mpath="./models/"):
    """pts: list of (bits, exempt, nb, err, diff)"""
    tag = f"{mname}|{dset}|{ck}|sub{sub}|pr{pr}"
    key = lambda p: f"{phase}|{mname}|{dset}|{ck}|sub{sub}|pr{pr}|b{p[0]}|ex{p[1]}|e{p[3]}|d{int(p[4])}|nb{p[2]}"
    todo = [p for p in pts if key(p) not in DONE_KEYS]
    need_ol = any(p[4] for p in pts) and (tag + "|diff") not in OL_DONE
    if not todo and not need_ol: print("skip", tag); return
    model, dq, dt, C, hist, cnt, lay, names = build(mname, dset, pr, sub=sub, mpath=mpath)
    with torch.no_grad():                                   # one plain pass clears the layer-record hook
        model(next(iter(dt))[0][:8].cuda())
    if need_ol:
        MODE["diff"] = True; openloop(model, dq, C, hist, tag + "|diff", pr); MODE["diff"] = False
    sync_datasets()
    for b, ex, nb, er, df in todo:
        run(phase, model, dt, C, cnt, mname, dset, pr, nb, b, sub=sub, exempt=ex, ck=ck, err=er, diff=df)
    free(model)

ERRS = (0.05, 0.075, 0.1, 0.15)
# E + D, ResNet18/CIFAR-10, 128-row arrays at PR = 128
pts = [(5, 1, N_MID, e, False) for e in ERRS] + [(6, 0, N_MID, e, False) for e in ERRS] \
    + [(5, 1, N_FULL, 0.05, False), (6, 0, N_FULL, 0.05, False), (5, 4, N_FULL, 0.05, False)] \
    + [(4, 0, N_MID, 0.0, True), (5, 0, N_MID, 0.0, True), (6, 0, N_MID, 0.0, True), (5, 0, N_FULL, 0.0, True), (5, 0, N_MID, 0.05, True)]
point12("E", R18, C10, 128, 128, pts)
# E + D, ResNet18/CIFAR-10, 256-row arrays at PR = 256
pts = [(6, 1, N_MID, e, False) for e in ERRS] + [(6, 1, N_FULL, 0.05, False)] \
    + [(5, 0, N_MID, 0.0, True), (6, 0, N_MID, 0.0, True), (6, 0, N_FULL, 0.0, True)]
point12("E", R18, C10, 256, 256, pts)
# D, VGG8/CIFAR-10 at PR = 128
point12("E", "vgg8", C10, 128, 128, [(4, 0, N_MID, 0.0, True), (5, 0, N_MID, 0.0, True)])

print("\nDONE. Send me S12_runs.csv, S12_openloop.csv and a zip of S12_preds.")

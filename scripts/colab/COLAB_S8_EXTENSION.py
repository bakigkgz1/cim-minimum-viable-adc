# === PAPER III — S8: STRUCTURAL EXTENSION (one Colab cell, T4 GPU) ===
# Converter: textbook b-bit (2^b codes, clamp [0, 2^b - 1]) everywhere, including the hardware path.
#   A  256-row subarray (functional): open-loop histograms at PR 128-256 and accuracy of 6/7-bit
#      converters at PR 160-256 (ResNet18/C10; VGG8/C10 and ResNet18/C100 at PR 256), full test,
#      also with the offset path unclipped
#   R  offset path: the dummy row/column that removes the input/weight offset is converted WITHOUT
#      clipping (both, row only, column only). If the offset path is what clips, a converter that
#      handles it separately (or digital offset correction) would let every signal column use fewer bits.
#      ResNet18/C10 PR 32-128 (4/5-bit), VGG8 and ResNet18/C100 at PR 72-128; open-loop tails split
#      into signal and offset entries; per-layer feedback with the offset path unclipped
#   F  fine code-noise sweep sigma = 0.115-0.13 LSB with the realized code-change fraction
#   H  hardware path with the textbook clamp patched into the simulator, matched levelOutput,
#      flash-type MLSA and SAR, 128-row (PR64/5, PR64/6, PR128/6, 7, 8) and 256-row
#      (PR256/6, 7, 8, 9) subarrays; full component breakdown (ADC share of energy, area, latency)
#
# Approximate run time on a T4 (setup 0-8 min; resume-safe, rerun after a disconnect):
#   H ~1.5 h | F ~25 min | R ~3 h | A ~2.5 h      Total ~7–7.5 h
# Outputs (Drive/NeuroSim15v_paper3/full_rerun_v1/): S8_runs.csv, S8_openloop.csv, S8_hist.csv,
#   S8_layers.csv, S8_hw.csv, S8_preds/*.npy, logs/hw_S8_*.log

RUN_H, RUN_F, RUN_R, RUN_A = True, True, True, True     # order: H first (fast), then F, R, A
N_FULL, N_MID, N_QUICK = -1, 32, 16

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
S8R = OUT / "S8_runs.csv"; S8O = OUT / "S8_openloop.csv"; S8H = OUT / "S8_hist.csv"
S8L = OUT / "S8_layers.csv"; S8W = OUT / "S8_hw.csv"
PRED = OUT / "S8_preds"; PRED.mkdir(exist_ok=True)
DONE_KEYS = {r["key"] for r in load_rows(S8R)}
L_DONE = {r["exp"] for r in load_rows(S8L)}
OL_DONE = {r["cfg"] for r in load_rows(S8O)}
HW_DONE = {r["tag"] for r in load_rows(S8W)}


def build(model_name, dataset, pr, sub=128, seed=1234):
    sys.argv = ["inference.py", "--dataset", dataset, "--model", model_name, "--hardware", "0", "--ppa", "0",
                "--num_batches", "16", "--batch_size", "64", "--calib_batch_size", "128", "--seed", str(seed),
                "--parallel_read", str(pr), "--adc_precision", "8", "--sub_array", f"[{sub},{sub}]",
                "--data_path", "./datasets/", "--model_path", "./models/",
                "--test_name", f"s8_{model_name}_{dataset}_{pr}_{sub}"]
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
    C = dict(stats=False, bits=8, only=None, probe=5, exempt=0, noise=0.0)
    hist, cnt, lay = {}, {"nelem": 0, "nalt": 0, "ndum": 0, "ndum_alt": 0, "nflip": 0}, {}
    hsig, hdum = {}, {}
    names = []
    for name, m in model.named_modules():
        if isinstance(m, macro.CIM) and hasattr(m, "_adc_quant"):
            names.append(name)
            orig = m._adc_quant
            def wrapped(self, x, _orig=orig, _n=name):
                if self._adc_quantizer is None or self._adc_quantizer._disabled or self._adc_quantizer._if_calib:
                    return _orig(x)
                if C["stats"]:
                    xr = x.detach().round().clamp(0, pr).long()
                    h = torch.bincount(xr.flatten(), minlength=pr + 1).cpu()
                    hist[_n] = hist[_n] + h if _n in hist else h
                    if xr.dim() == 3:                                 # split signal and offset (dummy) entries
                        hs = torch.bincount(xr[:, :-1, :-1].flatten(), minlength=pr + 1).cpu()
                        hsig[_n] = hsig[_n] + hs if _n in hsig else hs
                        hd = h - hs; hdum[_n] = hdum[_n] + hd if _n in hdum else hd
                    return x
                b = C["bits"] if (C["only"] is None or C["only"] == _n) else 8
                top = 2 ** b - 1
                y = x.detach()
                if C["noise"] > 0:                                   # code noise, clamped at zero
                    yn = torch.round(y + C["noise"] * torch.randn_like(y))
                    cnt["nflip"] += int((yn != y).sum().item()); y = yn.clamp(min=0)
                rec = lay.setdefault(_n, [0, 0, 0.0, 0])
                rec[0] += y.numel(); rec[1] += int((y >= 2 ** C["probe"]).sum().item())
                rec[2] += float(y.sum().item()); alt = int((y > top).sum().item()); rec[3] += alt
                cnt["nelem"] += y.numel(); cnt["nalt"] += alt
                q = y.clamp(0, top)
                if y.dim() == 3:                                      # (partitions, rows + dummy, cols + dummy)
                    dr, dc = y[:, -1, :], y[:, :-1, -1]
                    cnt["ndum"] += dr.numel() + dc.numel()
                    cnt["ndum_alt"] += int((dr > top).sum().item() + (dc > top).sum().item())
                    if C["exempt"]:
                        q = q.clone()
                        if C["exempt"] in (1, 2): q[:, -1, :] = y[:, -1, :]
                        if C["exempt"] in (1, 3): q[:, :, -1] = y[:, :, -1]
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


def run(phase, model, dt, C, cnt, mname, dset, pr, nb, bits, sub=128, exempt=0, noise=0.0):
    key = f"{phase}|{mname}|{dset}|sub{sub}|pr{pr}|b{bits}|ex{int(exempt)}|n{noise}|nb{nb}"
    if key in DONE_KEYS: print("skip", key); return
    C.update(bits=bits, only=None, probe=bits, exempt=exempt, noise=noise)
    for k in cnt: cnt[k] = 0
    torch.manual_seed(1234); torch.cuda.manual_seed_all(1234); t0 = time.time(); correct = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dt):
            if nb >= 0 and i >= nb: break
            correct.append((model(x.cuda()).argmax(1) == y.cuda()).cpu().numpy())
    c = np.concatenate(correct)
    fname = re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".npy"; np.save(PRED / fname, c)
    row = dict(ts=utc(), key=key, phase=phase, model=mname, dataset=dset, sub_array=sub, pr=pr, bits=bits,
               exempt_dummy=int(exempt), sigma_lsb=noise, n_images=c.size, acc=round(100.0 * c.mean(), 4),
               altered_frac=f"{cnt['nalt'] / max(cnt['nelem'], 1):.3e}",
               dummy_altered_frac=f"{cnt['ndum_alt'] / max(cnt['ndum'], 1):.3e}",
               code_change_frac=f"{cnt['nflip'] / max(cnt['nelem'], 1):.3e}",
               seconds=round(time.time() - t0, 1), pred_file=fname)
    append_row(S8R, row); DONE_KEYS.add(key); C.update(exempt=0, noise=0.0)
    print(f"S8 {phase} {mname}/{dset} sub{sub} PR{pr} {bits}b exempt={int(exempt)} σ={noise}: acc {row['acc']} "
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

R18, C10, C100 = "resnet18", "cifar10", "cifar100"

# ================================================================ H: hardware path, textbook clamp
if RUN_H:
    MAC = ROOT / "pytorch-quantization" / "pytorch_quantization" / "cim" / "modules" / "macro.py"
    MAC_ORIG = MAC.read_text()
    patched = MAC_ORIG.replace("adc_max = 2**self._adc_quantizer.num_bits\n", "adc_max = 2**self._adc_quantizer.num_bits - 1\n")
    patched = patched.replace("max=float(2**self._cim_args.adc_precision))", "max=float(2**self._cim_args.adc_precision - 1))")
    patched = patched.replace("max= float(2**self._cim_args.adc_precision))", "max= float(2**self._cim_args.adc_precision - 1))")
    patched = patched.replace("max=2**self._cim_args.adc_precision)", "max=2**self._cim_args.adc_precision - 1)")
    assert patched.count("num_bits - 1\n") == 1 and patched.count("adc_precision - 1") == 3, "textbook-clamp patch did not apply; send me macro.py lines 180-270"
    PARAM = ROOT / "NeuroSIM" / "Param.cpp"; BASE = DRIVE / "blok2d_Param_BASE.cpp"
    if not BASE.exists(): shutil.copy2(PARAM, BASE)
    def set_param(adc, sar):
        text = BASE.read_text()
        text, n1 = re.subn(r"levelOutput\s*=\s*\d+\s*;", f"levelOutput = {2**adc};", text, count=1)
        text, n2 = re.subn(r"SARADC\s*=\s*(true|false)\s*;", f"SARADC = {'true' if sar else 'false'};", text, count=1)
        assert n1 == 1 and n2 == 1
        PARAM.write_text(text)
        assert os.system(f"cd {ROOT/'NeuroSIM'} && make -j2 -s") == 0, "NeuroSIM build failed"
    def last(rx, t):
        m = list(re.finditer(rx, t)); return m[-1].group(1) if m else ""
    jobs = [(128, pr, b) for pr, b in ((64, 5), (64, 6), (128, 6), (128, 7), (128, 8))] + \
           [(256, 256, b) for b in (6, 7, 8, 9)]
    try:
        MAC.write_text(patched)
        for sar in (False, True):
            for sub, pr, adc in jobs:
                tag = f"S8_{'SAR' if sar else 'MLSA'}_sub{sub}_PR{pr}_ADC{adc}"
                if tag in HW_DONE: print("skip", tag); continue
                set_param(adc, sar); t0 = time.time(); logf = LOG / f"hw_{tag}.log"
                cmd = [sys.executable, "-u", str(ROOT / "inference.py"), "--dataset", "cifar10", "--model", "resnet18",
                       "--hardware", "1", "--ppa", "1", "--num_batches", "1", "--batch_size", "64",
                       "--parallel_read", str(pr), "--adc_precision", str(adc), "--sub_array", f"[{sub},{sub}]",
                       "--data_path", "./datasets/", "--model_path", "./models/", "--test_name", f"hw_{tag}"]
                with logf.open("w") as f:
                    p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
                t = logf.read_text(errors="ignore")
                if p.returncode != 0: print(t[-2500:]); raise RuntimeError(f"hardware run {tag} failed")
                num = r"\s*:?\s*([\d.eE+-]+)"
                row = dict(ts=utc(), tag=tag, arch="SAR" if sar else "MLSA", sub_array=sub, pr=pr, adc=adc, levelOutput=2 ** adc,
                           tops_w=last(r"Energy Efficiency TOPS/W \(Pipelined Process\):" + num, t),
                           fps=last(r"Throughput FPS \(Pipelined Process\):" + num, t),
                           tops=last(r"Throughput TOPS \(Pipelined Process\):" + num, t),
                           area=last(r"ChipArea" + num, t), array_area=last(r"Chip total CIM array" + num, t),
                           ic_area=last(r"Total IC Area on chip \(Global and Tile/PE local\):" + num, t),
                           adc_area=last(r"Total ADC \(or S/As and precharger for SRAM\) Area on chip" + num, t),
                           accum_area=last(r"Total Accumulation Circuits .*? on chip" + num, t),
                           other_area=last(r"Other Peripheries \(e\.g\. decoders, mux, switchmatrix, buffers, pooling and activation units\)" + num, t),
                           clock_ns=last(r"Chip clock period is:" + num, t),
                           cycle_ns=last(r"Chip pipeline-system-clock-cycle \(per image\) is:" + num, t),
                           energy_pJ=last(r"Chip pipeline-system readDynamicEnergy \(per image\) is:" + num, t),
                           leak_pJ=last(r"Chip pipeline-system leakage Energy \(per image\) is:" + num, t),
                           buffer_lat_ns=last(r"Chip pipeline-system buffer readLatency \(per image\) is:" + num, t),
                           ic_lat_ns=last(r"Chip pipeline-system ic readLatency \(per image\) is:" + num, t),
                           adc_lat_ns=last(r"ADC \(or S/As and precharger for SRAM\) readLatency is" + num, t),
                           accum_lat_ns=last(r"Accumulation Circuits .*? readLatency is" + num, t),
                           other_lat_ns=last(r"Other Peripheries .*? readLatency is" + num, t),
                           adc_E_pJ=last(r"ADC \(or S/As and precharger for SRAM\) readDynamicEnergy is" + num, t),
                           accum_E_pJ=last(r"Accumulation Circuits .*? readDynamicEnergy is" + num, t),
                           other_E_pJ=last(r"Other Peripheries .*? readDynamicEnergy is" + num, t),
                           seconds=round(time.time() - t0, 1))
                append_row(S8W, row); HW_DONE.add(tag)
                print("HW", tag, {k: row[k] for k in ("tops_w", "fps", "area", "adc_area", "adc_E_pJ", "energy_pJ", "cycle_ns")}, flush=True)
    finally:
        MAC.write_text(MAC_ORIG)                                   # restore the simulator source
        set_param(7, False)

# ================================================================ F: fine code-noise sweep
if RUN_F:
    for pr, bits in ((128, 6), (64, 5)):
        todo = [s for s in (0.115, 0.12, 0.125, 0.13) if f"F|{R18}|{C10}|sub128|pr{pr}|b{bits}|ex0|n{s}|nb{N_QUICK}" not in DONE_KEYS]
        if not todo: continue
        model, dq, dt, C, hist, cnt, lay, names = build(R18, C10, pr)
        for s in todo: run("F", model, dt, C, cnt, R18, C10, pr, N_QUICK, bits, noise=s)
        free(model)

# ================================================================ R: offset (dummy) path converted without clipping
# exempt codes: 1 = dummy row and column unclipped, 2 = dummy row only, 3 = dummy column only, 0 = reference
if RUN_R:
    R_PLAN = [(R18, C10, 72, [(5, 1, N_FULL), (5, 0, N_QUICK)]),
              (R18, C10, 80, [(5, 1, N_FULL), (5, 2, N_MID), (5, 3, N_MID), (5, 0, N_QUICK)]),
              (R18, C10, 88, [(5, 1, N_FULL)]), (R18, C10, 96, [(5, 1, N_FULL)]),
              (R18, C10, 128, [(5, 1, N_FULL), (4, 1, N_MID)]),
              (R18, C10, 32, [(4, 1, N_MID)]), (R18, C10, 48, [(4, 1, N_MID)]), (R18, C10, 64, [(4, 1, N_MID)]),
              ("vgg8", C10, 80, [(5, 1, N_FULL)]), ("vgg8", C10, 128, [(5, 1, N_FULL)]),
              (R18, C100, 72, [(5, 1, N_FULL)]), (R18, C100, 128, [(5, 1, N_FULL)])]
    for mname, dset, pr, pts in R_PLAN:
        tag = f"{mname}|{dset}|sub128|pr{pr}"
        exps = [f"R_pr80_all_exempt", "R_pr80_only_conv2_x.0_exempt"] if (mname, dset, pr) == (R18, C10, 80) else []
        todo = [(b, ex, nb) for b, ex, nb in pts if f"R|{mname}|{dset}|sub128|pr{pr}|b{b}|ex{ex}|n0.0|nb{nb}" not in DONE_KEYS]
        if not todo and tag in OL_DONE and all(x in L_DONE for x in exps): print("skip R", tag); continue
        model, dq, dt, C, hist, cnt, lay, names = build(mname, dset, pr)
        openloop(model, dq, C, hist, tag, pr); sync_datasets()
        for b, ex, nb in todo: run("R", model, dt, C, cnt, mname, dset, pr, nb, b, exempt=ex)
        if exps:
            layer_record("R_pr80_all_exempt", model, dt, C, hist, lay, names, 5, exempt=1)
            if "conv2_x.0.residual_function.0" in names:
                layer_record("R_pr80_only_conv2_x.0_exempt", model, dt, C, hist, lay, names, 5,
                             only="conv2_x.0.residual_function.0", exempt=1)
        free(model)

# ================================================================ A: 256-row subarray
if RUN_A:
    plan = [(R18, C10, 128, []), (R18, C10, 160, [(6, 0, N_FULL)]), (R18, C10, 192, [(6, 0, N_FULL)]),
            (R18, C10, 224, [(6, 0, N_FULL)]),
            (R18, C10, 256, [(6, 0, N_FULL), (7, 0, N_FULL), (5, 0, N_MID), (6, 1, N_FULL), (5, 1, N_MID)]),
            ("vgg8", C10, 256, [(6, 0, N_FULL), (7, 0, N_FULL), (6, 1, N_FULL)]),
            (R18, C100, 256, [(6, 0, N_FULL), (7, 0, N_FULL), (6, 1, N_FULL)])]
    for mname, dset, pr, pts in plan:
        tag = f"{mname}|{dset}|sub256|pr{pr}"
        todo = [(b, ex, nb) for b, ex, nb in pts if f"A|{mname}|{dset}|sub256|pr{pr}|b{b}|ex{ex}|n0.0|nb{nb}" not in DONE_KEYS]
        if tag in OL_DONE and not todo: print("skip", tag); continue
        model, dq, dt, C, hist, cnt, lay, names = build(mname, dset, pr, sub=256)
        openloop(model, dq, C, hist, tag, pr); sync_datasets()
        for b, ex, nb in todo: run("A", model, dt, C, cnt, mname, dset, pr, nb, b, sub=256, exempt=ex)
        free(model)

print("\nDONE. Send me S8_runs.csv, S8_openloop.csv, S8_hist.csv, S8_layers.csv, S8_hw.csv and a zip of S8_preds.")

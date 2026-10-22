# === PAPER3 FULL RERUN (flat — no nested quotes) ===
# All claim-critical experiments from scratch. Resume-safe.
#
# 1) Upload this file to Drive:
#      MyDrive/NeuroSim15v_paper3/COLAB_FULL_RERUN.py
# 2) In Colab (NeuroSim15v already set up):
#      %run /content/drive/MyDrive/NeuroSim15v_paper3/COLAB_FULL_RERUN.py
#
# Outputs under: Drive/NeuroSim15v_paper3/full_rerun_v1/
# If a phase CSV already exists with enough rows, that phase is SKIPPED.
# To force redo one phase: delete its CSV in full_rerun_v1/ then re-%run.
#
# ETA (T4, approximate):
#   P1 Blok1 Acc 512     ~2–4 h
#   P2 Clip grid         ~1–2 h
#   P3 C10 full Acc      ~2–4 h
#   P4 C100 full Acc     ~2–4 h  (patches Normalize to 0.5)
#   P5 VGG8              ~0.5 h
#   P6 HW+energy         ~2–3 h
#   P7 Multi-seed        ~1–2 h
#   P8 Variation         ~1–2 h
# Total ~12–20 h — leave overnight; resume after disconnect.

from __future__ import annotations

import csv
import hashlib
import os
import re
import statistics as st
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/content/NeuroSim15v")
DRIVE = Path("/content/drive/MyDrive/NeuroSim15v_paper3")
OUT = DRIVE / "full_rerun_v1"
LOG = OUT / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

assert ROOT.exists(), f"Missing {ROOT} — clone/setup NeuroSim15v first"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
# NeuroSim's package, not NVIDIA TensorRT pytorch_quantization
sys.path.insert(0, str(ROOT / "pytorch-quantization"))

# Force re-run specific phases even if CSV exists, e.g. FORCE = {"P3", "P6"}
FORCE: set[str] = set()

pq = ROOT / "pytorch-quantization"
assert pq.exists(), f"Missing {pq}"
rc = os.system(
    f'{sys.executable} -c "from pytorch_quantization.cim.modules import macro"'
)
if rc != 0:
    print("Installing NeuroSim pytorch-quantization editable...", flush=True)
    rc2 = os.system(f"{sys.executable} -m pip -q install -e {pq}")
    assert rc2 == 0, "pip install -e pytorch-quantization failed"
    rc3 = os.system(
        f'{sys.executable} -c "from pytorch_quantization.cim.modules import macro"'
    )
    assert rc3 == 0, "cim.macro still missing after pip install"

import torch
import torch.nn as nn
import pytorch_quantization.cim.modules.macro as macro
from inference import parse_args
from quantize import quantize_model
from dataset import get_cifar10, get_cifar100
from pytorch_quantization.utils import misc, make_path, hook
import dataset as dataset_mod

print("OK imports: torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def append_manifest(phase, status, detail=""):
    path = OUT / "manifest.csv"
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "phase", "status", "detail"])
        if write_header:
            w.writeheader()
        w.writerow({"ts": utc_now(), "phase": phase, "status": status, "detail": detail})
    print(f"[MANIFEST] {phase} {status} {detail}", flush=True)


def phase_done(csv_name: str, min_rows: int) -> bool:
    p = OUT / csv_name
    if not p.exists():
        return False
    with p.open() as f:
        n = sum(1 for _ in f) - 1
    return n >= min_rows


def write_csv(path: Path, rows: list[dict]):
    assert rows
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def install_clip_hooks(model):
    stats = {"nelem": 0, "nclip": 0, "nmax": 0.0}

    def make_wrapped(orig):
        def wrapped(self, inputs):
            if self._adc_quantizer is not None and not self._adc_quantizer._disabled:
                adc_max = float(2 ** int(self._adc_quantizer.num_bits))
                x = inputs.detach()
                ne = int(x.numel())
                nc = int((x >= adc_max).sum().item()) if ne else 0
                stats["nelem"] += ne
                stats["nclip"] += nc
                if ne:
                    stats["nmax"] = max(stats["nmax"], float(x.max().item()))
            return orig(inputs)

        return wrapped

    for m in model.modules():
        if isinstance(m, macro.CIM) and hasattr(m, "_adc_quant"):
            m._adc_quant = types.MethodType(make_wrapped(m._adc_quant), m)
    return stats


def run_eval(
    *,
    pr: int,
    adc: int,
    num_batches: int,
    seed: int = 1234,
    model_name: str = "resnet18",
    dataset: str = "cifar10",
    hardware: int = 0,
    want_clip: bool = False,
    read_noise: float = 0.0,
    rate_stuck_0: float = 0.0,
    rate_stuck_1: float = 0.0,
):
    """In-process Acc (+ optional clip). num_batches=-1 => full loader length."""
    sys.argv = [
        "inference.py",
        "--dataset", dataset,
        "--model", model_name,
        "--hardware", str(hardware),
        "--ppa", "0",
        "--num_batches", str(num_batches),
        "--batch_size", "64",
        "--calib_batch_size", "128",
        "--seed", str(seed),
        "--parallel_read", str(pr),
        "--adc_precision", str(adc),
        "--sub_array", "[128,128]",
        "--read_noise", str(read_noise),
        "--rate_stuck_0", str(rate_stuck_0),
        "--rate_stuck_1", str(rate_stuck_1),
        "--data_path", "./datasets/",
        "--model_path", "./models/",
        "--test_name", f"full_{dataset}_{model_name}_pr{pr}_adc{adc}_s{seed}",
    ]
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    criterion = nn.CrossEntropyLoss()
    args.data_path = args.data_path + args.dataset + "/"
    args.logdir = make_path.makepath(
        args,
        ["log_interval", "test_interval", "logdir", "epochs", "gpu", "ngpu", "debug", "data_path", "model_path"],
    )
    misc.ensure_dir(args.logdir)
    misc.logger.init(args.logdir, "full_rerun")
    args.logger = misc.logger.info

    if dataset == "cifar10":
        dq = get_cifar10(args.calib_batch_size, args.data_path, train=True, val=False)
        dt = get_cifar10(args.batch_size, args.data_path, train=False, val=True)
    elif dataset == "cifar100":
        dq = get_cifar100(args.calib_batch_size, args.data_path, train=True, val=False)
        dt = get_cifar100(args.batch_size, args.data_path, train=False, val=True)
    else:
        raise ValueError(dataset)

    hook.make_records(args)
    args.hook = True
    args.write_network = True
    model = quantize_model(args, criterion, dq, dt).to(torch.device(f"cuda:{args.gpu}")).eval()
    stats = install_clip_hooks(model) if want_clip else {"nelem": 0, "nclip": 0, "nmax": 0.0}

    nb = num_batches
    if nb < 0:
        nb = len(dt)

    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dt):
            if i >= nb:
                break
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()

    clip_frac = (stats["nclip"] / stats["nelem"]) if stats["nelem"] else float("nan")
    return {
        "pr": pr,
        "adc": adc,
        "model": model_name,
        "dataset": dataset,
        "seed": seed,
        "hardware": hardware,
        "read_noise": read_noise,
        "rate_stuck_0": rate_stuck_0,
        "rate_stuck_1": rate_stuck_1,
        "n_images": total,
        "acc": round(100.0 * correct / max(total, 1), 4),
        "clip_frac": float(f"{clip_frac:.8f}") if want_clip else "",
        "vmax_seen": round(stats["nmax"], 4) if want_clip else "",
        "adc_max": float(2 ** adc),
    }


def sha_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def set_lo(adc: int):
    PARAM = ROOT / "NeuroSIM" / "Param.cpp"
    BASE = DRIVE / "blok2d_Param_BASE.cpp"
    assert BASE.exists(), f"Missing {BASE}"
    lo = 2 ** adc
    text = BASE.read_text()
    new, n = re.subn(r"levelOutput\s*=\s*\d+\s*;", f"levelOutput = {lo};", text, count=1)
    assert n == 1, "levelOutput assign not found in BASE Param.cpp"
    PARAM.write_text(new)
    (LOG / f"Param_ADC{adc}_lo{lo}.cpp").write_text(new)
    assert os.system(f"cd {ROOT / 'NeuroSIM'} && make -j2 -s") == 0
    return lo, sha_text(new)


def last_match(rx: re.Pattern, text: str) -> str:
    ms = list(rx.finditer(text))
    return ms[-1].group(1) if ms else ""


def run_hw(pr: int, adc: int, tag: str) -> dict:
    lo, ph = set_lo(adc)
    log = LOG / f"hw_{tag}.log"
    cmd = [
        sys.executable, "-u", str(ROOT / "inference.py"),
        "--dataset", "cifar10", "--model", "resnet18",
        "--hardware", "1", "--ppa", "1",
        "--num_batches", "1", "--batch_size", "64",
        "--parallel_read", str(pr), "--adc_precision", str(adc),
        "--sub_array", "[128,128]",
        "--data_path", "./datasets/", "--model_path", "./models/",
        "--test_name", f"full_hw_{tag}",
    ]
    t0 = time.time()
    print(f"HW {tag} LO={lo} sha={ph}", flush=True)
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    text = log.read_text(errors="ignore")
    assert proc.returncode == 0, text[-3000:]
    RE_EFF = re.compile(r"Energy Efficiency TOPS/W \(Pipelined Process\):\s*([\d.eE+-]+)")
    RE_FPS = re.compile(r"Throughput FPS \(Pipelined Process\):\s*([\d.eE+-]+)")
    RE_TOPS = re.compile(r"Throughput TOPS \(Pipelined Process\):\s*([\d.eE+-]+)")
    RE_AREA = re.compile(r"ChipArea\s*:\s*([\d.eE+-]+)")
    RE_ADC_E = re.compile(r"ADC \(or S/As and precharger for SRAM\) readDynamicEnergy is\s*:\s*([\d.eE+-]+)")
    RE_ACC_E = re.compile(r"Accumulation Circuits.*?readDynamicEnergy is\s*:\s*([\d.eE+-]+)")
    RE_OTH_E = re.compile(r"Other Peripheries.*?readDynamicEnergy is\s*:\s*([\d.eE+-]+)")
    row = {
        "ts": utc_now(),
        "tag": tag,
        "pr": pr,
        "adc": adc,
        "levelOutput": lo,
        "param_sha": ph,
        "tops_w": last_match(RE_EFF, text),
        "fps": last_match(RE_FPS, text),
        "tops": last_match(RE_TOPS, text),
        "area": last_match(RE_AREA, text),
        "adc_E_pJ": last_match(RE_ADC_E, text),
        "accum_E_pJ": last_match(RE_ACC_E, text),
        "other_E_pJ": last_match(RE_OTH_E, text),
        "seconds": round(time.time() - t0, 1),
    }
    try:
        a, c, o = float(row["adc_E_pJ"]), float(row["accum_E_pJ"]), float(row["other_E_pJ"])
        s = a + c + o
        row["adc_E_frac"] = round(a / s, 4) if s else ""
        row["accum_E_frac"] = round(c / s, 4) if s else ""
        row["other_E_frac"] = round(o / s, 4) if s else ""
    except Exception:
        row["adc_E_frac"] = row["accum_E_frac"] = row["other_E_frac"] = ""
    print(row, flush=True)
    return row


def patch_cifar100_normalize_05():
    """Match Paper3 Blok4c: train Normalize(0.5), not CIFAR-100 stats."""
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader

    def get_cifar100_05(batch_size, data_root="/tmp/public_dataset/pytorch", train=True, val=True, **kwargs):
        data_root = os.path.expanduser(os.path.join(data_root, "cifar100-data"))
        kwargs.setdefault("num_workers", 4)
        ds = []
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        if train:
            train_data = datasets.CIFAR100(root=data_root, train=True, download=True, transform=tf)
            ds.append(DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs))
        if val:
            test_data = datasets.CIFAR100(root=data_root, train=False, download=True, transform=tf)
            ds.append(DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs))
        return ds[0] if len(ds) == 1 else ds

    dataset_mod.get_cifar100 = get_cifar100_05
    globals()["get_cifar100"] = get_cifar100_05
    print("Patched get_cifar100 -> Normalize(0.5)", flush=True)


# =====================================================================
# P1 — Blok1 Acc grid (512 images)
# =====================================================================
def phase_p1():
    name = "P1_blok1_acc.csv"
    corners = [(pr, adc) for adc in (5, 6, 8, 12) for pr in (16, 32, 64, 96, 128)]
    if "P1" not in FORCE and phase_done(name, len(corners)):
        append_manifest("P1", "SKIP", name)
        return
    append_manifest("P1", "START", f"n={len(corners)}")
    rows = []
    for pr, adc in corners:
        t0 = time.time()
        print(f"\nP1 PR={pr} ADC={adc}", flush=True)
        r = run_eval(pr=pr, adc=adc, num_batches=8, want_clip=False)
        r["ts"] = utc_now()
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(r, flush=True)
        write_csv(OUT / name, rows)  # checkpoint each corner
    append_manifest("P1", "DONE", name)


# =====================================================================
# P2 — Clip grid (dense)
# =====================================================================
def phase_p2():
    name = "P2_clip_grid.csv"
    corners = []
    for pr in (16, 32, 64, 72, 80, 88, 96, 128):
        corners.append((pr, 5))
    for adc in (6, 7, 8):
        corners.append((128, adc))
    for pr in (16, 32, 64, 96, 128):
        corners.append((pr, 8))
    # unique preserve order
    seen = set()
    uniq = []
    for c in corners:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    corners = uniq
    if "P2" not in FORCE and phase_done(name, len(corners)):
        append_manifest("P2", "SKIP", name)
        return
    append_manifest("P2", "START", f"n={len(corners)}")
    rows = []
    for pr, adc in corners:
        t0 = time.time()
        print(f"\nP2 clip PR={pr} ADC={adc}", flush=True)
        r = run_eval(pr=pr, adc=adc, num_batches=4, want_clip=True)
        r["ts"] = utc_now()
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(r, flush=True)
        write_csv(OUT / name, rows)
    append_manifest("P2", "DONE", name)


# =====================================================================
# P3 — CIFAR-10 full Acc
# =====================================================================
def phase_p3():
    name = "P3_c10_full_acc.csv"
    corners = [(64, 5), (96, 5), (128, 5), (128, 6), (128, 7), (128, 8)]
    if "P3" not in FORCE and phase_done(name, len(corners)):
        append_manifest("P3", "SKIP", name)
        return
    append_manifest("P3", "START", f"n={len(corners)}")
    rows = []
    for pr, adc in corners:
        t0 = time.time()
        print(f"\nP3 FULL PR={pr} ADC={adc}", flush=True)
        r = run_eval(pr=pr, adc=adc, num_batches=-1, want_clip=False)
        r["ts"] = utc_now()
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(r, flush=True)
        write_csv(OUT / name, rows)
    append_manifest("P3", "DONE", name)


# =====================================================================
# P4 — CIFAR-100 full Acc (Normalize 0.5)
# =====================================================================
def phase_p4():
    name = "P4_c100_full_acc.csv"
    corners = [(64, 5), (128, 5), (128, 6), (128, 8)]
    if "P4" not in FORCE and phase_done(name, len(corners)):
        append_manifest("P4", "SKIP", name)
        return
    patch_cifar100_normalize_05()
    # need model: resnet18 cifar100 checkpoint
    model_path = ROOT / "models" / "resnet18_cifar100.pth"
    if not model_path.exists():
        alt = DRIVE / "models" / "resnet18_cifar100.pth"
        if alt.exists():
            os.system(f"cp {alt} {model_path}")
    if not model_path.exists():
        append_manifest("P4", "FAIL", "missing resnet18_cifar100.pth")
        print("SKIP P4: no cifar100 checkpoint", flush=True)
        return
    append_manifest("P4", "START", f"n={len(corners)}")
    rows = []
    for pr, adc in corners:
        t0 = time.time()
        print(f"\nP4 C100 FULL PR={pr} ADC={adc}", flush=True)
        r = run_eval(pr=pr, adc=adc, num_batches=-1, dataset="cifar100", want_clip=False)
        r["ts"] = utc_now()
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(r, flush=True)
        write_csv(OUT / name, rows)
    append_manifest("P4", "DONE", name)


# =====================================================================
# P5 — VGG8 control + clip
# =====================================================================
def phase_p5():
    name = "P5_vgg8.csv"
    corners = [(64, 5), (64, 8), (128, 5), (128, 6), (128, 8)]
    if "P5" not in FORCE and phase_done(name, len(corners)):
        append_manifest("P5", "SKIP", name)
        return
    append_manifest("P5", "START", f"n={len(corners)}")
    rows = []
    for pr, adc in corners:
        t0 = time.time()
        print(f"\nP5 VGG8 PR={pr} ADC={adc}", flush=True)
        r = run_eval(pr=pr, adc=adc, num_batches=8, model_name="vgg8", want_clip=True)
        r["ts"] = utc_now()
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(r, flush=True)
        write_csv(OUT / name, rows)
    append_manifest("P5", "DONE", name)


# =====================================================================
# P6 — Matched HW + energy
# =====================================================================
def phase_p6():
    name = "P6_hw_energy.csv"
    corners = [
        ("PR64_ADC5", 64, 5),
        ("PR128_ADC5", 128, 5),
        ("PR64_ADC8", 64, 8),
        ("PR128_ADC8", 128, 8),
        ("PR128_ADC6", 128, 6),
        ("PR128_ADC7", 128, 7),
    ]
    if "P6" not in FORCE and phase_done(name, len(corners)):
        append_manifest("P6", "SKIP", name)
        return
    append_manifest("P6", "START", f"n={len(corners)}")
    rows = []
    for tag, pr, adc in corners:
        rows.append(run_hw(pr, adc, tag))
        write_csv(OUT / name, rows)
    set_lo(8)
    append_manifest("P6", "DONE", name)


# =====================================================================
# P7 — Multi-seed
# =====================================================================
def phase_p7():
    name = "P7_multiseed.csv"
    corners = [(64, 5), (128, 5), (128, 6)]
    seeds = [1234, 2345, 3456]
    need = len(corners) * len(seeds)
    if "P7" not in FORCE and phase_done(name, need):
        append_manifest("P7", "SKIP", name)
        return
    append_manifest("P7", "START", f"n={need}")
    rows = []
    for pr, adc in corners:
        for seed in seeds:
            t0 = time.time()
            print(f"\nP7 PR={pr} ADC={adc} seed={seed}", flush=True)
            r = run_eval(pr=pr, adc=adc, num_batches=8, seed=seed, want_clip=False)
            r["ts"] = utc_now()
            r["seconds"] = round(time.time() - t0, 1)
            rows.append(r)
            print(r, flush=True)
            write_csv(OUT / name, rows)
    print("\n=== P7 SUMMARY mean+/-std ===", flush=True)
    for pr, adc in corners:
        vals = [r["acc"] for r in rows if r["pr"] == pr and r["adc"] == adc]
        print(f"{pr}\t{adc}\t{st.mean(vals):.2f}\t{st.stdev(vals):.2f}\t{vals}", flush=True)
    append_manifest("P7", "DONE", name)


# =====================================================================
# P8 — Variation stress
# =====================================================================
def phase_p8():
    name = "P8_variation.csv"
    corners = [(64, 5), (128, 5), (128, 8)]
    stressors = [
        {"read_noise": 0.0, "rate_stuck_0": 0.0, "rate_stuck_1": 0.0, "label": "nominal"},
        {"read_noise": 0.1, "rate_stuck_0": 0.0, "rate_stuck_1": 0.0, "label": "rn0.1"},
        {"read_noise": 0.2, "rate_stuck_0": 0.0, "rate_stuck_1": 0.0, "label": "rn0.2"},
        {"read_noise": 0.0, "rate_stuck_0": 0.01, "rate_stuck_1": 0.01, "label": "stuck1pct"},
    ]
    need = len(corners) * len(stressors)
    if "P8" not in FORCE and phase_done(name, need):
        append_manifest("P8", "SKIP", name)
        return
    append_manifest("P8", "START", f"n={need}")
    rows = []
    for pr, adc in corners:
        for st_ in stressors:
            t0 = time.time()
            print(f"\nP8 PR={pr} ADC={adc} {st_['label']}", flush=True)
            r = run_eval(
                pr=pr,
                adc=adc,
                num_batches=8,
                want_clip=False,
                read_noise=st_["read_noise"],
                rate_stuck_0=st_["rate_stuck_0"],
                rate_stuck_1=st_["rate_stuck_1"],
            )
            r["label"] = st_["label"]
            r["ts"] = utc_now()
            r["seconds"] = round(time.time() - t0, 1)
            rows.append(r)
            print(r, flush=True)
            write_csv(OUT / name, rows)
    append_manifest("P8", "DONE", name)


# =====================================================================
# MAIN
# =====================================================================
print("=" * 70, flush=True)
print("PAPER3 FULL RERUN  OUT=", OUT, flush=True)
print("=" * 70, flush=True)

assert (ROOT / "models").exists(), "models/ missing"
assert (DRIVE / "blok2d_Param_BASE.cpp").exists(), "blok2d_Param_BASE.cpp missing on Drive"

for fn in (
    phase_p1,
    phase_p2,
    phase_p3,
    phase_p4,
    phase_p5,
    phase_p6,
    phase_p7,
    phase_p8,
):
    fn()

print("\n" + "=" * 70, flush=True)
print("FULL RERUN COMPLETE (or resumed)", flush=True)
print("Folder:", OUT, flush=True)
print("Paste manifest.csv + key CSVs summary back to chat.", flush=True)
print("=" * 70, flush=True)

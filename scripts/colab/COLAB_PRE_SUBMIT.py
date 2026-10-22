# PAPER3 pre-submit — one Colab cell (self-contained, resume-safe).
# Does not need a prior FULL_RERUN cell.
#
# Default ON (cheap, hardware=0):
#   1) Layer-wise clip_frac  -> P2b_clip_by_layer.csv
#   2) PR64/ADC8 full test   -> append P3_c10_full_acc.csv  (drops Table II †)
#   3) Stuck-at 10 maps      -> P8b_stuck_multiseed.csv     (median + min/max)
#
# Default OFF (slow analog path; time-box one day if you flip it):
#   4) hardware=1 read_noise -> P8b_readnoise_hw1.csv
#
# Existing P2_clip_grid.csv is aggregated only (no tensor dumps) — layer CSV
# requires this pass. Paste Drive CSVs back to chat when done.

from __future__ import annotations

import csv
import os
import shutil
import statistics as st
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")

# --- flags ---
DO_LAYER = True
DO_PR64_ADC8_FULL = True
DO_STUCK10 = True
DO_HW1_READNOISE = False  # analog path; set True only with a one-day time box

ROOT = Path("/content/NeuroSim15v")
DRIVE = Path("/content/drive/MyDrive/NeuroSim15v_paper3")
OUT = DRIVE / "full_rerun_v1"
OUT.mkdir(parents=True, exist_ok=True)

if not ROOT.exists():
    print("Cloning NeuroSim 2DInferenceV1.5-dev ->", ROOT, flush=True)
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "-b",
            "2DInferenceV1.5-dev",
            "https://github.com/neurosim/NeuroSim.git",
            str(ROOT),
        ]
    )

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pytorch-quantization"))

models_dst = ROOT / "models"
models_dst.mkdir(parents=True, exist_ok=True)
models_src = DRIVE / "models"
if models_src.exists():
    for p in models_src.glob("*.pth"):
        dst = models_dst / p.name
        if not dst.exists():
            print("copy model", p.name, flush=True)
            shutil.copy2(p, dst)
print("models:", sorted(x.name for x in models_dst.glob("*.pth")), flush=True)

pq = ROOT / "pytorch-quantization"
rc = os.system(
    f'{sys.executable} -c "from pytorch_quantization.cim.modules import macro"'
)
if rc != 0:
    print("Installing NeuroSim pytorch-quantization ...", flush=True)
    rc2 = os.system(f"{sys.executable} -m pip -q install -e {pq}")
    assert rc2 == 0
    rc3 = os.system(
        f'{sys.executable} -c "from pytorch_quantization.cim.modules import macro"'
    )
    assert rc3 == 0

import torch
import torch.nn as nn
import pytorch_quantization.cim.modules.macro as macro
from inference import parse_args
from quantize import quantize_model
from dataset import get_cifar10
from pytorch_quantization.utils import misc, make_path, hook

print("OK imports cuda", torch.cuda.is_available(), "ROOT", ROOT, flush=True)
print(
    "FLAGS  DO_LAYER=%s  DO_PR64_ADC8_FULL=%s  DO_STUCK10=%s  DO_HW1_READNOISE=%s"
    % (DO_LAYER, DO_PR64_ADC8_FULL, DO_STUCK10, DO_HW1_READNOISE),
    flush=True,
)
if DO_HW1_READNOISE:
    print(
        "WARNING: hardware=1 simulates the analog MAC+ADC path. "
        "Budget a calendar day for 3 corners x sigma in {0,0.1,0.2} at N=512.",
        flush=True,
    )


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_rows(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
    assert rows
    preferred = [
        "ts",
        "model",
        "dataset",
        "pr",
        "adc",
        "layer",
        "layer_type",
        "clip_frac",
        "vmax",
        "nclip",
        "nelem",
        "in_channels",
        "out_channels",
        "rows_per_partition",
        "num_arrays_y",
        "kernel_size",
        "in_features",
        "seed",
        "hardware",
        "read_noise",
        "rate_stuck_0",
        "rate_stuck_1",
        "n_images",
        "acc",
        "clip_frac_agg",
        "vmax_seen",
        "adc_max",
        "label",
        "seconds",
    ]
    keys = []
    seen = set()
    for k in preferred:
        if any(k in r for r in rows) and k not in seen:
            keys.append(k)
            seen.add(k)
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def cim_layer_fields(name, m, pr):
    cim = getattr(m, "_cim_args", None)
    in_ch = getattr(m, "in_channels", None)
    out_ch = getattr(m, "out_channels", None)
    if in_ch is None:
        in_ch = getattr(m, "in_features", "")
    if out_ch is None:
        out_ch = getattr(m, "out_features", "")
    ksz = getattr(m, "kernel_size", "")
    if isinstance(ksz, (tuple, list)):
        ksz = "x".join(str(x) for x in ksz)
    rows_pr = pr
    n_y = ""
    in_feat = ""
    if cim is not None:
        rows_pr = getattr(cim, "parallel_read", pr)
        n_y = getattr(cim, "num_arrays_y", "")
        shp = getattr(cim, "weight2d_shape", None)
        if shp:
            in_feat = shp[0]
    return {
        "layer": name,
        "layer_type": type(m).__name__,
        "in_channels": in_ch,
        "out_channels": out_ch,
        "kernel_size": ksz,
        "rows_per_partition": rows_pr,
        "num_arrays_y": n_y,
        "in_features": in_feat,
    }


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


def install_layer_clip_hooks(model, pr):
    per = {}
    meta = {}

    def make_wrapped(orig, layer_name):
        def wrapped(self, inputs):
            if self._adc_quantizer is not None and not self._adc_quantizer._disabled:
                adc_max = float(2 ** int(self._adc_quantizer.num_bits))
                x = inputs.detach()
                ne = int(x.numel())
                nc = int((x >= adc_max).sum().item()) if ne else 0
                d = per.setdefault(
                    layer_name, {"nelem": 0, "nclip": 0, "nmax": 0.0}
                )
                d["nelem"] += ne
                d["nclip"] += nc
                if ne:
                    d["nmax"] = max(d["nmax"], float(x.max().item()))
            return orig(inputs)

        return wrapped

    for name, m in model.named_modules():
        if isinstance(m, macro.CIM) and hasattr(m, "_adc_quant"):
            meta[name] = cim_layer_fields(name, m, pr)
            m._adc_quant = types.MethodType(make_wrapped(m._adc_quant, name), m)
    return per, meta


def run_eval(
    *,
    pr,
    adc,
    num_batches,
    seed=1234,
    model_name="resnet18",
    hardware=0,
    want_clip=False,
    want_layer_clip=False,
    read_noise=0.0,
    rate_stuck_0=0.0,
    rate_stuck_1=0.0,
):
    sys.argv = [
        "inference.py",
        "--dataset", "cifar10",
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
        "--test_name", "presub_%s_pr%s_adc%s_s%s_hw%s"
        % (model_name, pr, adc, seed, hardware),
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
    misc.logger.init(args.logdir, "pre_submit")
    args.logger = misc.logger.info
    dq = get_cifar10(args.calib_batch_size, args.data_path, train=True, val=False)
    dt = get_cifar10(args.batch_size, args.data_path, train=False, val=True)
    hook.make_records(args)
    args.hook = True
    args.write_network = True
    model = quantize_model(args, criterion, dq, dt).to(
        torch.device("cuda:%s" % args.gpu)
    ).eval()

    layer_per, layer_meta = {}, {}
    stats = {"nelem": 0, "nclip": 0, "nmax": 0.0}
    if want_layer_clip:
        layer_per, layer_meta = install_layer_clip_hooks(model, pr)
    elif want_clip:
        stats = install_clip_hooks(model)

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

    if want_layer_clip:
        for d in layer_per.values():
            stats["nelem"] += d["nelem"]
            stats["nclip"] += d["nclip"]
            stats["nmax"] = max(stats["nmax"], d["nmax"])

    clip_frac = (stats["nclip"] / stats["nelem"]) if stats["nelem"] else float("nan")
    layer_rows = []
    if want_layer_clip:
        for name, d in layer_per.items():
            cf = (d["nclip"] / d["nelem"]) if d["nelem"] else float("nan")
            row = {
                "model": model_name,
                "dataset": "cifar10",
                "pr": pr,
                "adc": adc,
                "seed": seed,
                "n_images": total,
                "clip_frac": float("%.8f" % cf) if d["nelem"] else "",
                "vmax": round(d["nmax"], 4),
                "nclip": d["nclip"],
                "nelem": d["nelem"],
                "adc_max": float(2 ** adc),
            }
            row.update(layer_meta.get(name, {"layer": name}))
            layer_rows.append(row)
        layer_rows.sort(key=lambda r: -float(r["clip_frac"] or 0))

    return {
        "pr": pr,
        "adc": adc,
        "model": model_name,
        "dataset": "cifar10",
        "seed": seed,
        "hardware": hardware,
        "read_noise": read_noise,
        "rate_stuck_0": rate_stuck_0,
        "rate_stuck_1": rate_stuck_1,
        "n_images": total,
        "acc": round(100.0 * correct / max(total, 1), 4),
        "clip_frac": float("%.8f" % clip_frac) if (want_clip or want_layer_clip) else "",
        "vmax_seen": round(stats["nmax"], 4) if (want_clip or want_layer_clip) else "",
        "adc_max": float(2 ** adc),
        "layers": layer_rows,
    }


print("=" * 70, flush=True)
print("PRE-SUBMIT  OUT=", OUT, flush=True)
print("=" * 70, flush=True)

# ---------------------------------------------------------------------------
# 1) Layer-wise clip_frac (N=256 = 4 batches). Highest ROI.
# ---------------------------------------------------------------------------
if DO_LAYER:
    path_layer = OUT / "P2b_clip_by_layer.csv"
    layer_rows = load_rows(path_layer)
    layer_done = {
        (r.get("model", "resnet18"), int(r["pr"]), int(r["adc"])) for r in layer_rows
    }
    print("LAYER already", sorted(layer_done), flush=True)

    # ResNet cliff + recovery; VGG8 same corners to explain 0.0864 vs 0.567.
    layer_jobs = [
        ("resnet18", 64, 5),
        ("resnet18", 96, 5),
        ("resnet18", 128, 5),
        ("resnet18", 128, 6),
        ("vgg8", 128, 5),
        ("vgg8", 128, 6),
    ]
    for model_name, pr, adc in layer_jobs:
        key = (model_name, pr, adc)
        if key in layer_done:
            print("SKIP LAYER %s PR=%s ADC=%s" % key, flush=True)
            continue
        t0 = time.time()
        print("LAYER %s PR=%s ADC=%s N=256" % key, flush=True)
        r = run_eval(
            pr=pr,
            adc=adc,
            num_batches=4,
            model_name=model_name,
            want_layer_clip=True,
        )
        sec = round(time.time() - t0, 1)
        print(
            "  agg acc=%s clip_frac=%s vmax=%s layers=%s sec=%s"
            % (r["acc"], r["clip_frac"], r["vmax_seen"], len(r["layers"]), sec),
            flush=True,
        )
        top = r["layers"][:8]
        for row in top:
            print(
                "    %s  clip=%.6f vmax=%s Cin=%s Cout=%s rows=%s"
                % (
                    row.get("layer"),
                    float(row["clip_frac"] or 0),
                    row["vmax"],
                    row.get("in_channels"),
                    row.get("out_channels"),
                    row.get("rows_per_partition"),
                ),
                flush=True,
            )
        for row in r["layers"]:
            row["ts"] = utc_now()
            row["seconds"] = sec
            row["acc"] = r["acc"]
            row["clip_frac_agg"] = r["clip_frac"]
            layer_rows.append(row)
        layer_done.add(key)
        save_rows(path_layer, layer_rows)
    print("LAYER DONE", path_layer, flush=True)
else:
    print("SKIP phase LAYER (DO_LAYER=False)", flush=True)

# ---------------------------------------------------------------------------
# 2) PR64/ADC8 full test — removes the last Table II dagger.
# ---------------------------------------------------------------------------
if DO_PR64_ADC8_FULL:
    path_p3 = OUT / "P3_c10_full_acc.csv"
    p3_rows = load_rows(path_p3)
    p3_done = {(int(r["pr"]), int(r["adc"])) for r in p3_rows}
    print("P3 already", sorted(p3_done), flush=True)
    if (64, 8) in p3_done:
        print("SKIP PR64/ADC8 FULL", flush=True)
    else:
        t0 = time.time()
        print("FULL PR=64 ADC=8 n=10000", flush=True)
        r = run_eval(pr=64, adc=8, num_batches=-1, want_clip=True)
        r["ts"] = utc_now()
        r["seconds"] = round(time.time() - t0, 1)
        r.pop("layers", None)
        p3_rows.append(r)
        print(r, flush=True)
        save_rows(path_p3, p3_rows)
    print("PR64/ADC8 FULL DONE", path_p3, flush=True)
else:
    print("SKIP phase PR64/ADC8 FULL", flush=True)

# ---------------------------------------------------------------------------
# 3) Stuck-at 1% per state, 10 maps, N=512. Resume skips existing seeds.
# ---------------------------------------------------------------------------
STUCK_SEEDS = (1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 10123)

if DO_STUCK10:
    path_stuck = OUT / "P8b_stuck_multiseed.csv"
    stuck_rows = load_rows(path_stuck)
    stuck_done = {
        (int(r["pr"]), int(r["adc"]), int(r["seed"])) for r in stuck_rows
    }
    print("STUCK already", sorted(stuck_done), flush=True)
    for pr, adc in ((64, 5), (128, 5), (128, 8)):
        for seed in STUCK_SEEDS:
            if (pr, adc, seed) in stuck_done:
                print("SKIP STUCK PR=%s ADC=%s seed=%s" % (pr, adc, seed), flush=True)
                continue
            t0 = time.time()
            print("STUCK PR=%s ADC=%s seed=%s" % (pr, adc, seed), flush=True)
            r = run_eval(
                pr=pr,
                adc=adc,
                num_batches=8,
                seed=seed,
                rate_stuck_0=0.01,
                rate_stuck_1=0.01,
            )
            r["label"] = "stuck1pct"
            r["ts"] = utc_now()
            r["seconds"] = round(time.time() - t0, 1)
            r.pop("layers", None)
            stuck_rows.append(r)
            print(r, flush=True)
            save_rows(path_stuck, stuck_rows)

    print("=== stuck median / min / max (do not report mean+/-std) ===", flush=True)
    for pr, adc in ((64, 5), (128, 5), (128, 8)):
        vals = [
            float(r["acc"])
            for r in stuck_rows
            if int(r["pr"]) == pr and int(r["adc"]) == adc
        ]
        if vals:
            print(
                pr,
                adc,
                "n=%s" % len(vals),
                "median=%.4f" % st.median(vals),
                "min=%.4f" % min(vals),
                "max=%.4f" % max(vals),
                vals,
                flush=True,
            )
        else:
            print(pr, adc, "empty", flush=True)
    print("STUCK10 DONE", path_stuck, flush=True)
else:
    print("SKIP phase STUCK10", flush=True)

# ---------------------------------------------------------------------------
# 4) Optional hardware=1 read_noise (off by default).
# ---------------------------------------------------------------------------
if DO_HW1_READNOISE:
    path_rn = OUT / "P8b_readnoise_hw1.csv"
    rn_rows = load_rows(path_rn)
    rn_done = {
        (int(r["pr"]), int(r["adc"]), float(r["read_noise"])) for r in rn_rows
    }
    print("HW1 already", sorted(rn_done), flush=True)
    for pr, adc in ((64, 5), (128, 5), (128, 8)):
        for rn in (0.0, 0.1, 0.2):
            if (pr, adc, rn) in rn_done:
                print("SKIP HW1 PR=%s ADC=%s rn=%s" % (pr, adc, rn), flush=True)
                continue
            t0 = time.time()
            print("HW1 PR=%s ADC=%s rn=%s N=512" % (pr, adc, rn), flush=True)
            r = run_eval(
                pr=pr,
                adc=adc,
                num_batches=8,
                hardware=1,
                read_noise=rn,
            )
            r["label"] = "hw1_rn%s" % rn
            r["ts"] = utc_now()
            r["seconds"] = round(time.time() - t0, 1)
            r.pop("layers", None)
            rn_rows.append(r)
            print(r, flush=True)
            save_rows(path_rn, rn_rows)
    print("HW1 DONE", path_rn, flush=True)
else:
    print("SKIP phase HW1 read_noise (DO_HW1_READNOISE=False)", flush=True)

print("=" * 70, flush=True)
print("PRE-SUBMIT COMPLETE", flush=True)
print("Paste back to chat:", flush=True)
print("  ", OUT / "P2b_clip_by_layer.csv", flush=True)
print("  ", OUT / "P3_c10_full_acc.csv", flush=True)
print("  ", OUT / "P8b_stuck_multiseed.csv", flush=True)
if DO_HW1_READNOISE:
    print("  ", OUT / "P8b_readnoise_hw1.csv", flush=True)
print("=" * 70, flush=True)

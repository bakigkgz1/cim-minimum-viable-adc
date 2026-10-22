# Layer-wise clip_frac — one Colab cell, self-contained, resume-safe.
# P2_clip_grid.csv is aggregated over CIM layers and has no tensor dumps, so
# this 256-image pass is required (not a re-parse). Writes P2b_clip_by_layer.csv.
#
# Corners: ResNet18 (64,5)/(96,5)/(128,5)/(128,6) and VGG8 (128,5)/(128,6).
# For the combined pre-submit cell (also PR64/ADC8 full + stuck-10) use
# COLAB_PRE_SUBMIT.py instead.

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")

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

pq = ROOT / "pytorch-quantization"
rc = os.system(
    f'{sys.executable} -c "from pytorch_quantization.cim.modules import macro"'
)
if rc != 0:
    rc2 = os.system(f"{sys.executable} -m pip -q install -e {pq}")
    assert rc2 == 0

import torch
import torch.nn as nn
import pytorch_quantization.cim.modules.macro as macro
from inference import parse_args
from quantize import quantize_model
from dataset import get_cifar10
from pytorch_quantization.utils import misc, make_path, hook

print("OK cuda", torch.cuda.is_available(), flush=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_rows(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
    keys = [
        "ts",
        "model",
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
        "n_images",
        "acc",
        "clip_frac_agg",
        "adc_max",
        "seconds",
    ]
    extra = []
    seen = set(keys)
    for r in rows:
        for k in r:
            if k not in seen:
                extra.append(k)
                seen.add(k)
    fieldnames = keys + extra
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


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


def run_layer_eval(pr, adc, model_name):
    sys.argv = [
        "inference.py",
        "--dataset", "cifar10",
        "--model", model_name,
        "--hardware", "0",
        "--ppa", "0",
        "--num_batches", "4",
        "--batch_size", "64",
        "--calib_batch_size", "128",
        "--seed", "1234",
        "--parallel_read", str(pr),
        "--adc_precision", str(adc),
        "--sub_array", "[128,128]",
        "--data_path", "./datasets/",
        "--model_path", "./models/",
        "--test_name", "layer_%s_pr%s_adc%s" % (model_name, pr, adc),
    ]
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    criterion = nn.CrossEntropyLoss()
    args.data_path = args.data_path + args.dataset + "/"
    args.logdir = make_path.makepath(
        args,
        ["log_interval", "test_interval", "logdir", "epochs", "gpu", "ngpu", "debug", "data_path", "model_path"],
    )
    misc.ensure_dir(args.logdir)
    misc.logger.init(args.logdir, "layer_clip")
    args.logger = misc.logger.info
    dq = get_cifar10(args.calib_batch_size, args.data_path, train=True, val=False)
    dt = get_cifar10(args.batch_size, args.data_path, train=False, val=True)
    hook.make_records(args)
    args.hook = True
    args.write_network = True
    model = quantize_model(args, criterion, dq, dt).to(
        torch.device("cuda:%s" % args.gpu)
    ).eval()
    per, meta = install_layer_clip_hooks(model, pr)
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dt):
            if i >= 4:
                break
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
    nclip = sum(d["nclip"] for d in per.values())
    nelem = sum(d["nelem"] for d in per.values())
    vmax_agg = max((d["nmax"] for d in per.values()), default=0.0)
    agg = (nclip / nelem) if nelem else float("nan")
    rows = []
    for name, d in per.items():
        cf = (d["nclip"] / d["nelem"]) if d["nelem"] else float("nan")
        row = {
            "model": model_name,
            "pr": pr,
            "adc": adc,
            "n_images": total,
            "acc": round(100.0 * correct / max(total, 1), 4),
            "clip_frac": float("%.8f" % cf) if d["nelem"] else "",
            "vmax": round(d["nmax"], 4),
            "nclip": d["nclip"],
            "nelem": d["nelem"],
            "clip_frac_agg": float("%.8f" % agg) if nelem else "",
            "adc_max": float(2 ** adc),
        }
        row.update(meta.get(name, {"layer": name}))
        rows.append(row)
    rows.sort(key=lambda r: -float(r["clip_frac"] or 0))
    return rows, {
        "acc": round(100.0 * correct / max(total, 1), 4),
        "clip_frac": float("%.8f" % agg) if nelem else "",
        "vmax": round(vmax_agg, 4),
        "n_layers": len(rows),
    }


path = OUT / "P2b_clip_by_layer.csv"
rows = load_rows(path)
done = {(r.get("model", "resnet18"), int(r["pr"]), int(r["adc"])) for r in rows}
print("already", sorted(done), flush=True)

jobs = [
    ("resnet18", 64, 5),
    ("resnet18", 96, 5),
    ("resnet18", 128, 5),
    ("resnet18", 128, 6),
    ("vgg8", 128, 5),
    ("vgg8", 128, 6),
]
for model_name, pr, adc in jobs:
    key = (model_name, pr, adc)
    if key in done:
        print("SKIP", key, flush=True)
        continue
    t0 = time.time()
    print("RUN", key, flush=True)
    layer_rows, agg = run_layer_eval(pr, adc, model_name)
    sec = round(time.time() - t0, 1)
    print("  agg", agg, "sec", sec, flush=True)
    for row in layer_rows[:8]:
        print(
            "   ",
            row["layer"],
            "clip",
            row["clip_frac"],
            "vmax",
            row["vmax"],
            "Cin",
            row["in_channels"],
            "Cout",
            row["out_channels"],
            flush=True,
        )
    for row in layer_rows:
        row["ts"] = utc_now()
        row["seconds"] = sec
        rows.append(row)
    done.add(key)
    save_rows(path, rows)

print("DONE", path, flush=True)
print("Paste P2b_clip_by_layer.csv back to chat.", flush=True)

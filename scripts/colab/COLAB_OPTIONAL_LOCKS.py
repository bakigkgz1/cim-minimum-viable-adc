# PAPER3 optional locks — one Colab cell (setup + VGG8 full + stuck 3-seed)
# Resume-safe. Does not need a prior FULL_RERUN cell or a Drive .py upload.

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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_rows(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
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
    pr,
    adc,
    num_batches,
    seed=1234,
    model_name="resnet18",
    hardware=0,
    want_clip=False,
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
        "--rate_stuck_0", str(rate_stuck_0),
        "--rate_stuck_1", str(rate_stuck_1),
        "--data_path", "./datasets/",
        "--model_path", "./models/",
        "--test_name", "opt_%s_pr%s_adc%s_s%s" % (model_name, pr, adc, seed),
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
    misc.logger.init(args.logdir, "optional_locks")
    args.logger = misc.logger.info
    dq = get_cifar10(args.calib_batch_size, args.data_path, train=True, val=False)
    dt = get_cifar10(args.batch_size, args.data_path, train=False, val=True)
    hook.make_records(args)
    args.hook = True
    args.write_network = True
    model = quantize_model(args, criterion, dq, dt).to(torch.device("cuda:%s" % args.gpu)).eval()
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
        "dataset": "cifar10",
        "seed": seed,
        "hardware": hardware,
        "rate_stuck_0": rate_stuck_0,
        "rate_stuck_1": rate_stuck_1,
        "n_images": total,
        "acc": round(100.0 * correct / max(total, 1), 4),
        "clip_frac": float("%.8f" % clip_frac) if want_clip else "",
        "vmax_seen": round(stats["nmax"], 4) if want_clip else "",
        "adc_max": float(2 ** adc),
    }


print("=" * 70, flush=True)
print("OPTIONAL LOCKS  OUT=", OUT, flush=True)
print("=" * 70, flush=True)

path_vgg = OUT / "P5b_vgg8_full.csv"
vgg_rows = load_rows(path_vgg)
vgg_done = {(int(r["pr"]), int(r["adc"])) for r in vgg_rows}
print("VGG8 already", sorted(vgg_done), flush=True)

for pr, adc in ((128, 5), (128, 6)):
    if (pr, adc) in vgg_done:
        print("SKIP VGG8 FULL PR=%s ADC=%s" % (pr, adc), flush=True)
        continue
    t0 = time.time()
    print("VGG8 FULL PR=%s ADC=%s" % (pr, adc), flush=True)
    r = run_eval(
        pr=pr,
        adc=adc,
        num_batches=-1,
        model_name="vgg8",
        want_clip=True,
    )
    r["ts"] = utc_now()
    r["seconds"] = round(time.time() - t0, 1)
    vgg_rows.append(r)
    print(r, flush=True)
    save_rows(path_vgg, vgg_rows)

print("VGG8 FULL DONE", path_vgg, flush=True)

path_stuck = OUT / "P8b_stuck_multiseed.csv"
stuck_rows = load_rows(path_stuck)
stuck_done = {
    (int(r["pr"]), int(r["adc"]), int(r["seed"])) for r in stuck_rows
}
print("STUCK already", sorted(stuck_done), flush=True)

for pr, adc in ((64, 5), (128, 5), (128, 8)):
    for seed in (1234, 2345, 3456):
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
        stuck_rows.append(r)
        print(r, flush=True)
        save_rows(path_stuck, stuck_rows)

print("=== stuck mean+/-std ===", flush=True)
for pr, adc in ((64, 5), (128, 5), (128, 8)):
    vals = [
        float(r["acc"])
        for r in stuck_rows
        if int(r["pr"]) == pr and int(r["adc"]) == adc
    ]
    if len(vals) >= 2:
        print(pr, adc, round(st.mean(vals), 2), round(st.stdev(vals), 2), vals, flush=True)
    else:
        print(pr, adc, vals, flush=True)

print("=" * 70, flush=True)
print("OPTIONAL LOCKS COMPLETE", flush=True)
print("Paste P5b_vgg8_full.csv + P8b_stuck_multiseed.csv back to chat.", flush=True)
print("=" * 70, flush=True)

# Optional: conductance read_noise on the hardware=1 accuracy path.
# Acc-path (hardware=0) never calls ADC_output, so sigma does nothing there.
#
# Self-contained. Analog MAC is slow — time-box one calendar day.
# 3 corners x sigma in {0.0, 0.1, 0.2} x N=512. Resume-safe.
# Writes Drive/NeuroSim15v_paper3/full_rerun_v1/P8b_readnoise_hw1.csv
#
# Prefer COLAB_PRE_SUBMIT.py with DO_HW1_READNOISE = True if you also want
# layer-clip / PR64-ADC8-full / stuck-10 in the same session.

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")

ROOT = Path("/content/NeuroSim15v")
DRIVE = Path("/content/drive/MyDrive/NeuroSim15v_paper3")
OUT = DRIVE / "full_rerun_v1"
OUT.mkdir(parents=True, exist_ok=True)

print(
    "hardware=1 analog path. 9 runs (3 corners x 3 sigma) at N=512. "
    "If a run hangs past ~2 h, skip that corner and keep the honest IV-G note.",
    flush=True,
)

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
    assert rows
    keys = list(rows[0].keys())
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


def run_eval(*, pr, adc, read_noise, num_batches=8, seed=1234):
    sys.argv = [
        "inference.py",
        "--dataset", "cifar10",
        "--model", "resnet18",
        "--hardware", "1",
        "--ppa", "0",
        "--num_batches", str(num_batches),
        "--batch_size", "64",
        "--calib_batch_size", "128",
        "--seed", str(seed),
        "--parallel_read", str(pr),
        "--adc_precision", str(adc),
        "--sub_array", "[128,128]",
        "--read_noise", str(read_noise),
        "--data_path", "./datasets/",
        "--model_path", "./models/",
        "--test_name", "hw1_pr%s_adc%s_rn%s" % (pr, adc, read_noise),
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
    misc.logger.init(args.logdir, "hw1_readnoise")
    args.logger = misc.logger.info
    dq = get_cifar10(args.calib_batch_size, args.data_path, train=True, val=False)
    dt = get_cifar10(args.batch_size, args.data_path, train=False, val=True)
    hook.make_records(args)
    args.hook = True
    args.write_network = True
    model = quantize_model(args, criterion, dq, dt).to(
        torch.device("cuda:%s" % args.gpu)
    ).eval()
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dt):
            if i >= num_batches:
                break
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
    return {
        "pr": pr,
        "adc": adc,
        "model": "resnet18",
        "dataset": "cifar10",
        "seed": seed,
        "hardware": 1,
        "read_noise": read_noise,
        "n_images": total,
        "acc": round(100.0 * correct / max(total, 1), 4),
        "adc_max": float(2 ** adc),
        "label": "hw1_rn%s" % read_noise,
    }


path = OUT / "P8b_readnoise_hw1.csv"
rows = load_rows(path)
done = {(int(r["pr"]), int(r["adc"]), float(r["read_noise"])) for r in rows}
print("already", sorted(done), flush=True)

for pr, adc in ((64, 5), (128, 5), (128, 8)):
    for rn in (0.0, 0.1, 0.2):
        if (pr, adc, rn) in done:
            print("SKIP PR=%s ADC=%s rn=%s" % (pr, adc, rn), flush=True)
            continue
        t0 = time.time()
        print("HW1 PR=%s ADC=%s rn=%s N=512" % (pr, adc, rn), flush=True)
        r = run_eval(pr=pr, adc=adc, read_noise=rn)
        r["ts"] = utc_now()
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(r, flush=True)
        save_rows(path, rows)

print("DONE", path, flush=True)
print("If this finished: IV-G gets a real variation result. If not: keep the honest note.", flush=True)

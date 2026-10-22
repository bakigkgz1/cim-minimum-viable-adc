# === PAPER3 REMAINING RUNS (flat — no nested quotes) ===
# Madde2: multi-seed | Madde3: PR96 full Acc
# Upload this file to Drive NeuroSim15v_paper3/ then in Colab:
#   %run /content/drive/MyDrive/NeuroSim15v_paper3/COLAB_REMAINING_RUNS.py
#
# Or paste entire file as one cell.

import os, sys, csv, time, statistics as st, re, subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/content/NeuroSim15v")
DRIVE = Path("/content/drive/MyDrive/NeuroSim15v_paper3")
assert ROOT.exists(), "NeuroSim15v missing"
os.chdir(ROOT)
DRIVE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

rc = os.system(f'{sys.executable} -c "import pytorch_quantization"')
if rc != 0:
    assert os.system(f"{sys.executable} -m pip -q install -e {ROOT / 'pytorch-quantization'}") == 0

import torch
import torch.nn as nn
from inference import parse_args
from quantize import quantize_model
from dataset import get_cifar10
from pytorch_quantization.utils import misc, make_path, hook

# ---------------------------------------------------------------------------
# Madde 2 — multi-seed
# ---------------------------------------------------------------------------
def run_acc(pr, adc, seed, num_batches, model_name="resnet18"):
    sys.argv = [
        "inference.py",
        "--dataset", "cifar10",
        "--model", model_name,
        "--hardware", "0",
        "--ppa", "0",
        "--num_batches", str(num_batches),
        "--batch_size", "64",
        "--calib_batch_size", "128",
        "--seed", str(seed),
        "--parallel_read", str(pr),
        "--adc_precision", str(adc),
        "--sub_array", "[128,128]",
        "--data_path", "./datasets/",
        "--model_path", "./models/",
        "--test_name", f"run_pr{pr}_adc{adc}_s{seed}_nb{num_batches}",
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
        [
            "log_interval",
            "test_interval",
            "logdir",
            "epochs",
            "gpu",
            "ngpu",
            "debug",
            "data_path",
            "model_path",
        ],
    )
    misc.ensure_dir(args.logdir)
    misc.logger.init(args.logdir, "remain_log")
    args.logger = misc.logger.info
    dq = get_cifar10(args.calib_batch_size, args.data_path, train=True, val=False)
    dt = get_cifar10(args.batch_size, args.data_path, train=False, val=True)
    hook.make_records(args)
    args.hook = True
    args.write_network = True
    model = quantize_model(args, criterion, dq, dt).to(torch.device(f"cuda:{args.gpu}")).eval()
    correct = total = 0
    # num_batches=-1 => full loader
    with torch.no_grad():
        for i, (x, y) in enumerate(dt):
            if num_batches >= 0 and i >= num_batches:
                break
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "pr": pr,
        "adc": adc,
        "seed": seed,
        "n_images": total,
        "acc": round(100.0 * correct / total, 4),
    }


print("=" * 60, flush=True)
print("MADDE 2: multi-seed (3 corners x 3 seeds x 512)", flush=True)
print("=" * 60, flush=True)

SEEDS = [1234, 2345, 3456]
CORNERS = [(64, 5), (128, 5), (128, 6)]
ms_rows = []
ms_out = DRIVE / "blok5_multiseed.csv"

for pr, adc in CORNERS:
    for seed in SEEDS:
        t0 = time.time()
        print(f"\n=== PR={pr} ADC={adc} seed={seed} ===", flush=True)
        row = run_acc(pr, adc, seed, num_batches=8)
        row["seconds"] = round(time.time() - t0, 1)
        ms_rows.append(row)
        print(row, flush=True)

with ms_out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ms_rows[0].keys()))
    w.writeheader()
    w.writerows(ms_rows)

print("\n=== SUMMARY mean+/-std ===", flush=True)
print("pr\tadc\tmean\tstd\tvals", flush=True)
for pr, adc in CORNERS:
    vals = [r["acc"] for r in ms_rows if r["pr"] == pr and r["adc"] == adc]
    mu = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{pr}\t{adc}\t{mu:.2f}\t{sd:.2f}\t{vals}", flush=True)
print("DONE multiseed ->", ms_out, flush=True)

# ---------------------------------------------------------------------------
# Madde 3 — PR96/ADC5 full Acc
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("MADDE 3: PR96/ADC5 full Acc (num_batches=-1)", flush=True)
print("=" * 60, flush=True)

t0 = time.time()
# Use subprocess inference for full test (same as prior full Acc locks)
log = DRIVE / "pr96_adc5_full_acc.log"
cmd = [
    sys.executable,
    "-u",
    str(ROOT / "inference.py"),
    "--dataset",
    "cifar10",
    "--model",
    "resnet18",
    "--hardware",
    "0",
    "--ppa",
    "0",
    "--num_batches",
    "-1",
    "--batch_size",
    "64",
    "--parallel_read",
    "96",
    "--adc_precision",
    "5",
    "--sub_array",
    "[128,128]",
    "--data_path",
    "./datasets/",
    "--model_path",
    "./models/",
    "--test_name",
    "pr96_adc5_full",
]
print("CMD:", " ".join(cmd), flush=True)
with log.open("w") as f:
    proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
text = log.read_text(errors="ignore")
RE_ACC = re.compile(r"Test Accuracy of the model on\s+(\d+)\s+test images:\s*([0-9.]+)\s*%")
hits = list(RE_ACC.finditer(text))
assert proc.returncode == 0 and hits, text[-2500:]
n_img, acc = int(hits[-1].group(1)), float(hits[-1].group(2))
row = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "pr": 96,
    "adc": 5,
    "n": n_img,
    "acc": acc,
    "seconds": round(time.time() - t0, 1),
}
out3 = DRIVE / "pr96_adc5_full_acc.csv"
with out3.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    w.writeheader()
    w.writerow(row)
print(row, flush=True)
print("DONE PR96 full ->", out3, flush=True)
print("\nALL REMAINING RUNS COMPLETE", flush=True)
print("Paste SUMMARY + PR96 row back to chat.", flush=True)

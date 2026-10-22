# === Madde2: multi-seed Acc (3 corners x 3 seeds, N=512) ===
# Paste this entire cell into Colab. No nested r''' strings.
import os, sys, csv, time, statistics as st
from pathlib import Path
from datetime import datetime, timezone
import torch
import torch.nn as nn

ROOT = Path("/content/NeuroSim15v")
DRIVE = Path("/content/drive/MyDrive/NeuroSim15v_paper3")
os.chdir(ROOT)
DRIVE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

rc = os.system(f'{sys.executable} -c "import pytorch_quantization"')
if rc != 0:
    assert os.system(f"{sys.executable} -m pip -q install -e {ROOT / 'pytorch-quantization'}") == 0
assert os.system(f'{sys.executable} -c "import pytorch_quantization"') == 0

from inference import parse_args
from quantize import quantize_model
from dataset import get_cifar10
from pytorch_quantization.utils import misc, make_path, hook

NUM_BATCHES = 8  # 512 images
SEEDS = [1234, 2345, 3456]
CORNERS = [
    (64, 5),
    (128, 5),
    (128, 6),
]
OUT = DRIVE / "blok5_multiseed.csv"


def run_corner(pr, adc, seed):
    sys.argv = [
        "inference.py",
        "--dataset", "cifar10",
        "--model", "resnet18",
        "--hardware", "0",
        "--ppa", "0",
        "--num_batches", str(NUM_BATCHES),
        "--batch_size", "64",
        "--calib_batch_size", "128",
        "--seed", str(seed),
        "--parallel_read", str(pr),
        "--adc_precision", str(adc),
        "--sub_array", "[128,128]",
        "--data_path", "./datasets/",
        "--model_path", "./models/",
        "--test_name", f"ms_pr{pr}_adc{adc}_s{seed}",
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
    misc.logger.init(args.logdir, "ms_log")
    args.logger = misc.logger.info
    dq = get_cifar10(args.calib_batch_size, args.data_path, train=True, val=False)
    dt = get_cifar10(args.batch_size, args.data_path, train=False, val=True)
    hook.make_records(args)
    args.hook = True
    args.write_network = True
    model = quantize_model(args, criterion, dq, dt).to(torch.device(f"cuda:{args.gpu}")).eval()
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dt):
            if i >= NUM_BATCHES:
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


rows = []
for pr, adc in CORNERS:
    for seed in SEEDS:
        t0 = time.time()
        print(f"\n=== PR={pr} ADC={adc} seed={seed} ===", flush=True)
        row = run_corner(pr, adc, seed)
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(row, flush=True)

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\n=== SUMMARY mean+/-std ===", flush=True)
print("pr\tadc\tmean\tstd\tvals", flush=True)
for pr, adc in CORNERS:
    vals = [r["acc"] for r in rows if r["pr"] == pr and r["adc"] == adc]
    mu = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{pr}\t{adc}\t{mu:.2f}\t{sd:.2f}\t{vals}", flush=True)
print("DONE ->", OUT, flush=True)

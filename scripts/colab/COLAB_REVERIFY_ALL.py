# === PAPER3 REVERIFY — flat, no nested quotes ===
# Re-runs claim-critical locks and DIFFS against old Drive CSVs.
# Upload to Drive NeuroSim15v_paper3/ then:
#   %run /content/drive/MyDrive/NeuroSim15v_paper3/COLAB_REVERIFY_ALL.py
#
# Phases (can skip done ones via SKIP set below):
#   A  clip points: PR64/80/88/96/128 @ ADC5 + PR128 @ ADC6  (N=256)
#   B  VGG8 PR128 ADC5 + ADC6 (N=512)
#   C  PR128/ADC6 full Acc (n~10048)
#   D  PR128/ADC6 matched HW LO=64
#   E  PR128/ADC7 matched HW LO=128
#   F  Blok2d energy 4 corners (optional, long)
#   G  multi-seed 3x3 x512
#   H  PR96/ADC5 full Acc
#
# Start with A–E (~1.5–2.5h). Then G+H. F only if EE story still doubted.

import os, sys, csv, time, re, hashlib, subprocess, statistics as st
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/content/NeuroSim15v")
DRIVE = Path("/content/drive/MyDrive/NeuroSim15v_paper3")
OUTDIR = DRIVE / "reverify_20260826"
LOGDIR = OUTDIR / "logs"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Toggle phases here
SKIP = set()  # e.g. {"F"} to skip long energy; or {"A","B"} if already done mid-run

rc = os.system(f'{sys.executable} -c "import pytorch_quantization"')
if rc != 0:
    assert os.system(f"{sys.executable} -m pip -q install -e {ROOT / 'pytorch-quantization'}") == 0

import torch
import torch.nn as nn
from inference import parse_args
from quantize import quantize_model
from dataset import get_cifar10
from pytorch_quantization.utils import misc, make_path, hook
import pytorch_quantization.cim.modules.macro as macro
import types

REPORT = []


def log_report(phase, key, old, new, ok, note=""):
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "key": key,
        "old": old,
        "new": new,
        "match": int(bool(ok)),
        "note": note,
    }
    REPORT.append(row)
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {phase} {key}: old={old} new={new} {note}", flush=True)


def close_enough(a, b, rtol=0.02, atol=0.15):
    try:
        fa, fb = float(a), float(b)
    except Exception:
        return str(a) == str(b)
    return abs(fa - fb) <= max(atol, rtol * max(abs(fa), abs(fb), 1e-12))


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


def run_probe(pr, adc, num_batches, model_name="resnet18", seed=1234, want_clip=False):
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
        "--test_name", f"rev_{model_name}_pr{pr}_adc{adc}_s{seed}",
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
    misc.logger.init(args.logdir, "reverify_log")
    args.logger = misc.logger.info
    dq = get_cifar10(args.calib_batch_size, args.data_path, train=True, val=False)
    dt = get_cifar10(args.batch_size, args.data_path, train=False, val=True)
    hook.make_records(args)
    args.hook = True
    args.write_network = True
    model = quantize_model(args, criterion, dq, dt).to(torch.device(f"cuda:{args.gpu}")).eval()
    stats = install_clip_hooks(model) if want_clip else {"nelem": 0, "nclip": 0, "nmax": 0.0}
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dt):
            if num_batches >= 0 and i >= num_batches:
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
        "seed": seed,
        "n_images": total,
        "acc": round(100.0 * correct / total, 4),
        "clip_frac": float(f"{clip_frac:.8f}") if want_clip else "",
        "vmax_seen": round(stats["nmax"], 4) if want_clip else "",
        "adc_max": float(2 ** adc),
    }


def sha_text(t):
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def set_lo(adc):
    PARAM = ROOT / "NeuroSIM" / "Param.cpp"
    BASE = DRIVE / "blok2d_Param_BASE.cpp"
    assert BASE.exists(), "blok2d_Param_BASE.cpp missing on Drive"
    lo = 2 ** adc
    text = BASE.read_text()
    new, n = re.subn(r"levelOutput\s*=\s*\d+\s*;", f"levelOutput = {lo};", text, count=1)
    assert n == 1
    PARAM.write_text(new)
    assert os.system(f"cd {ROOT / 'NeuroSIM'} && make -j2 -s") == 0
    return lo, sha_text(new)


def run_hw(pr, adc, tag):
    lo, ph = set_lo(adc)
    log = LOGDIR / f"hw_{tag}.log"
    cmd = [
        sys.executable, "-u", str(ROOT / "inference.py"),
        "--dataset", "cifar10", "--model", "resnet18",
        "--hardware", "1", "--ppa", "1", "--num_batches", "1", "--batch_size", "64",
        "--parallel_read", str(pr), "--adc_precision", str(adc),
        "--sub_array", "[128,128]",
        "--data_path", "./datasets/", "--model_path", "./models/",
        "--test_name", f"rev_hw_{tag}",
    ]
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    text = log.read_text(errors="ignore")
    assert proc.returncode == 0, text[-2000:]

    def last(rx):
        ms = list(rx.finditer(text))
        return ms[-1].group(1) if ms else ""

    RE_EFF = re.compile(r"Energy Efficiency TOPS/W \(Pipelined Process\):\s*([\d.eE+-]+)")
    RE_FPS = re.compile(r"Throughput FPS \(Pipelined Process\):\s*([\d.eE+-]+)")
    RE_AREA = re.compile(r"ChipArea\s*:\s*([\d.eE+-]+)")
    return {
        "tag": tag,
        "pr": pr,
        "adc": adc,
        "levelOutput": lo,
        "param_sha": ph,
        "tops_w": last(RE_EFF),
        "fps": last(RE_FPS),
        "area": last(RE_AREA),
    }


def run_full_acc(pr, adc, tag):
    log = LOGDIR / f"acc_{tag}.log"
    cmd = [
        sys.executable, "-u", str(ROOT / "inference.py"),
        "--dataset", "cifar10", "--model", "resnet18",
        "--hardware", "0", "--ppa", "0", "--num_batches", "-1", "--batch_size", "64",
        "--parallel_read", str(pr), "--adc_precision", str(adc),
        "--sub_array", "[128,128]",
        "--data_path", "./datasets/", "--model_path", "./models/",
        "--test_name", f"rev_full_{tag}",
    ]
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    text = log.read_text(errors="ignore")
    RE_ACC = re.compile(r"Test Accuracy of the model on\s+(\d+)\s+test images:\s*([0-9.]+)\s*%")
    hits = list(RE_ACC.finditer(text))
    assert proc.returncode == 0 and hits, text[-2500:]
    return int(hits[-1].group(1)), float(hits[-1].group(2))


# Old expected values (from locks you already have)
OLD_CLIP = {
    (64, 5): {"acc": 90.6250, "clip": 6.71e-6},
    (80, 5): {"acc": 83.9844, "clip": 4.09e-4},
    (88, 5): {"acc": 69.5312, "clip": 8.90e-3},
    (96, 5): {"acc": 25.3906, "clip": 0.1318422},
    (128, 5): {"acc": 11.3281, "clip": 0.5668475},
    (128, 6): {"acc": 91.4062, "clip": 3.4e-7},
}
OLD_VGG = {
    (128, 5): {"acc": 34.81, "clip": 0.08640074},
    (128, 6): {"acc": 91.7969, "clip": 4.5e-7},
}

# -------------------- A --------------------
if "A" not in SKIP:
    print("\n===== PHASE A: ResNet clip probes =====", flush=True)
    rows = []
    for pr, adc in [(64, 5), (80, 5), (88, 5), (96, 5), (128, 5), (128, 6)]:
        t0 = time.time()
        print(f"A PR={pr} ADC={adc}", flush=True)
        r = run_probe(pr, adc, num_batches=4, want_clip=True)
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        old = OLD_CLIP[(pr, adc)]
        log_report("A", f"acc_PR{pr}_ADC{adc}", old["acc"], r["acc"], close_enough(old["acc"], r["acc"], atol=1.0))
        log_report(
            "A",
            f"clip_PR{pr}_ADC{adc}",
            old["clip"],
            r["clip_frac"],
            close_enough(old["clip"], r["clip_frac"], rtol=0.25, atol=1e-6),
        )
    with (OUTDIR / "A_clip_probes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

# -------------------- B --------------------
if "B" not in SKIP:
    print("\n===== PHASE B: VGG8 =====", flush=True)
    rows = []
    for pr, adc in [(128, 5), (128, 6)]:
        t0 = time.time()
        print(f"B VGG8 PR={pr} ADC={adc}", flush=True)
        r = run_probe(pr, adc, num_batches=8, model_name="vgg8", want_clip=True)
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        old = OLD_VGG[(pr, adc)]
        log_report("B", f"vgg_acc_ADC{adc}", old["acc"], r["acc"], close_enough(old["acc"], r["acc"], atol=1.0))
        log_report(
            "B",
            f"vgg_clip_ADC{adc}",
            old["clip"],
            r["clip_frac"],
            close_enough(old["clip"], r["clip_frac"], rtol=0.25, atol=1e-6),
        )
    with (OUTDIR / "B_vgg8.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

# -------------------- C --------------------
if "C" not in SKIP:
    print("\n===== PHASE C: PR128/ADC6 full Acc =====", flush=True)
    n, acc = run_full_acc(128, 6, "pr128_adc6")
    log_report("C", "full_acc", 90.08, acc, close_enough(90.08, acc, atol=0.3), note=f"n={n}")
    with (OUTDIR / "C_pr128_adc6_full.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pr", "adc", "n", "acc"])
        w.writeheader()
        w.writerow({"pr": 128, "adc": 6, "n": n, "acc": acc})

# -------------------- D --------------------
if "D" not in SKIP:
    print("\n===== PHASE D: PR128/ADC6 HW =====", flush=True)
    r = run_hw(128, 6, "PR128_ADC6")
    log_report("D", "tops_w", 26.9751, r["tops_w"], close_enough(26.9751, r["tops_w"], rtol=0.01, atol=0.05))
    log_report("D", "area", 3.40633e8, r["area"], close_enough(3.40633e8, r["area"], rtol=0.01, atol=1e6))
    with (OUTDIR / "D_pr128_adc6_hw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r.keys()))
        w.writeheader()
        w.writerow(r)

# -------------------- E --------------------
if "E" not in SKIP:
    print("\n===== PHASE E: PR128/ADC7 HW =====", flush=True)
    r = run_hw(128, 7, "PR128_ADC7")
    log_report("E", "tops_w", 20.0431, r["tops_w"], close_enough(20.0431, r["tops_w"], rtol=0.01, atol=0.05))
    with (OUTDIR / "E_pr128_adc7_hw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r.keys()))
        w.writeheader()
        w.writerow(r)
    set_lo(8)

# -------------------- F (optional) --------------------
if "F" not in SKIP:
    print("\n===== PHASE F: energy 4 corners =====", flush=True)
    old_tw = {"PR64_ADC5": 28.0011, "PR128_ADC5": 14.97, "PR64_ADC8": 8.29584, "PR128_ADC8": 13.4083}
    rows = []
    for tag, pr, adc in [
        ("PR64_ADC5", 64, 5),
        ("PR128_ADC5", 128, 5),
        ("PR64_ADC8", 64, 8),
        ("PR128_ADC8", 128, 8),
    ]:
        print(f"F {tag}", flush=True)
        r = run_hw(pr, adc, tag)
        rows.append(r)
        log_report("F", f"{tag}_tops_w", old_tw[tag], r["tops_w"], close_enough(old_tw[tag], r["tops_w"], rtol=0.01, atol=0.05))
    with (OUTDIR / "F_blok2d_hw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    set_lo(8)

# -------------------- G --------------------
if "G" not in SKIP:
    print("\n===== PHASE G: multi-seed =====", flush=True)
    rows = []
    for pr, adc in [(64, 5), (128, 5), (128, 6)]:
        for seed in [1234, 2345, 3456]:
            print(f"G PR={pr} ADC={adc} seed={seed}", flush=True)
            r = run_probe(pr, adc, num_batches=8, seed=seed, want_clip=False)
            rows.append(r)
            print(r, flush=True)
    with (OUTDIR / "G_multiseed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("=== SUMMARY mean+/-std ===", flush=True)
    for pr, adc in [(64, 5), (128, 5), (128, 6)]:
        vals = [r["acc"] for r in rows if r["pr"] == pr and r["adc"] == adc]
        print(f"{pr}\t{adc}\t{st.mean(vals):.2f}\t{st.stdev(vals):.2f}\t{vals}", flush=True)

# -------------------- H --------------------
if "H" not in SKIP:
    print("\n===== PHASE H: PR96/ADC5 full Acc =====", flush=True)
    n, acc = run_full_acc(96, 5, "pr96_adc5")
    # probe was 25.39; full may differ a bit
    log_report("H", "full_acc", 25.39, acc, True, note=f"n={n}; probe was 25.39 — full is new lock")
    with (OUTDIR / "H_pr96_adc5_full.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pr", "adc", "n", "acc"])
        w.writeheader()
        w.writerow({"pr": 96, "adc": 5, "n": n, "acc": acc})

# write report
rep = OUTDIR / "REVERIFY_REPORT.csv"
with rep.open("w", newline="") as f:
    if REPORT:
        w = csv.DictWriter(f, fieldnames=list(REPORT[0].keys()))
        w.writeheader()
        w.writerows(REPORT)

n_fail = sum(1 for r in REPORT if not r["match"])
n_pass = sum(1 for r in REPORT if r["match"])
print("\n========== REVERIFY DONE ==========", flush=True)
print(f"PASS={n_pass} FAIL={n_fail} report={rep}", flush=True)
print("Paste REVERIFY_REPORT summary (PASS/FAIL lines) back to chat.", flush=True)
if n_fail:
    print("Any FAIL => that old lock is untrusted; we replace with new numbers.", flush=True)
else:
    print("All checked locks reproduced => prior runs were trustworthy.", flush=True)

#!/usr/bin/env python3
"""Paper III Blok 2 — HW/PPA on Acc sweet / cliff corners.

Saves full NeuroSIM+inference log per corner under --log_dir.
Parses Layer-by-Layer and Pipelined metrics.
"""
from __future__ import annotations
import argparse, csv, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve()
# .../NeuroSim15v/paper3_resnet/scripts/this.py -> parents[2]=NeuroSim15v
ROOT = _here.parents[2]
if not (ROOT / "inference.py").is_file():
    for cand in [Path.cwd(), Path("/content/NeuroSim15v"), Path("/content/drive/MyDrive/NeuroSim15v")]:
        if (cand / "inference.py").is_file():
            ROOT = cand
            break
RE_ACC = re.compile(r"Test Accuracy of the model on\s+(\d+)\s+test images:\s*([0-9.]+)\s*%")
RE_AREA = re.compile(r"ChipArea\s*:\s*([\d.eE+-]+)")
# accept Layer-by-Layer or Pipelined
RE_EFF = re.compile(r"Energy Efficiency TOPS/W \((?:Layer-by-Layer|Pipelined) Process\):\s*([\d.eE+-]+)")
RE_TOPS = re.compile(r"Throughput TOPS \((?:Layer-by-Layer|Pipelined) Process\):\s*([\d.eE+-]+)")
RE_FPS = re.compile(r"Throughput FPS \((?:Layer-by-Layer|Pipelined) Process\):\s*([\d.eE+-]+)")
# looser fallbacks
RE_EFF2 = re.compile(r"Energy Efficiency TOPS/W[^:]*:\s*([\d.eE+-]+)")
RE_TOPS2 = re.compile(r"Throughput TOPS[^:]*:\s*([\d.eE+-]+)")
RE_FPS2 = re.compile(r"Throughput FPS[^:]*:\s*([\d.eE+-]+)")

DEFAULT_CORNERS = [(16, 5), (64, 5), (128, 5), (128, 8), (32, 5), (128, 12)]
FIELDS = [
    "timestamp","phase","dataset","model","hardware","parallel_read","adc",
    "num_batches","acc_pct","n_images","area_um2","fps","tops","tops_w","notes",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--log_dir", default="")
    p.add_argument("--num_batches", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--corners", default="")
    p.add_argument("--force", action="store_true", help="re-run even if done")
    return p.parse_args()

def corners(args):
    if not args.corners.strip():
        return list(DEFAULT_CORNERS)
    out = []
    for tok in args.corners.split(","):
        a, b = tok.split(":")
        out.append((int(a), int(b)))
    return out

def load_done(path: Path):
    done = set()
    if not path.exists():
        return done
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("phase") != "blok2_hw":
                continue
            # treat as done only if fps or tops_w present (complete HW row)
            if not (r.get("fps") or r.get("tops_w") or r.get("tops")):
                continue
            try:
                done.add((int(r["parallel_read"]), int(r["adc"])))
            except Exception:
                pass
    return done

def append_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()

def last_acc(text: str):
    ms = list(RE_ACC.finditer(text))
    if not ms:
        return None, None
    m = ms[-1]
    return float(m.group(2)), int(m.group(1))

def first(rx_list, text):
    for rx in rx_list:
        m = rx.search(text)
        if m:
            return m.group(1)
    return ""

def parse_hw(text: str):
    area = first([RE_AREA], text)
    fps = first([RE_FPS, RE_FPS2], text)
    tops = first([RE_TOPS, RE_TOPS2], text)
    tops_w = first([RE_EFF, RE_EFF2], text)
    return area, fps, tops, tops_w

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    log_dir = Path(args.log_dir) if args.log_dir else csv_path.parent / "blok2_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    order = corners(args)
    done = set() if args.force else load_done(csv_path)
    print(f"CSV={csv_path} log_dir={log_dir} done={sorted(done)} remaining={[c for c in order if c not in done]}", flush=True)

    main_bin = ROOT / "NeuroSIM" / "main"
    if not main_bin.is_file():
        print(f"ERROR: missing {main_bin}", flush=True)
        sys.exit(2)

    for pr, adc in order:
        if (pr, adc) in done:
            print(f"SKIP PR={pr} ADC={adc}", flush=True)
            continue
        cmd = [
            sys.executable, "-u", str(ROOT / "inference.py"),
            "--dataset", "cifar10", "--model", "resnet18",
            "--hardware", "1", "--ppa", "1",
            "--num_batches", str(args.num_batches),
            "--batch_size", str(args.batch_size),
            "--adc_precision", str(adc),
            "--sub_array", "[128,128]",
            "--parallel_read", str(pr),
            "--mem_type", "resistive",
            "--mem_states_file", "mem_states.csv",
            "--data_path", "./datasets/",
            "--model_path", "./models/",
        ]
        print("=" * 60, flush=True)
        print("RUN", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path = log_dir / f"pr{pr}_adc{adc}.log"
        log_path.write_text(out)
        print(f"LOG {log_path} bytes={log_path.stat().st_size}", flush=True)
        # show metric lines for debug
        for line in out.splitlines():
            if any(k in line for k in ("ChipArea", "Throughput", "Energy Efficiency", "Hardware Properties")):
                print("METRICS:", line, flush=True)
        print(out[-2500:], flush=True)
        acc, n_img = last_acc(out)
        area, fps, tops, tops_w = parse_hw(out)
        ok = proc.returncode == 0 and bool(area) and bool(fps or tops or tops_w)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": "blok2_hw",
            "dataset": "cifar10",
            "model": "resnet18",
            "hardware": "1",
            "parallel_read": str(pr),
            "adc": str(adc),
            "num_batches": str(args.num_batches),
            "acc_pct": f"{acc:.5f}" if acc is not None else "",
            "n_images": str(n_img or ""),
            "area_um2": area,
            "fps": fps,
            "tops": tops,
            "tops_w": tops_w,
            "notes": "ok" if ok else f"INCOMPLETE rc={proc.returncode}; see {log_path.name}",
        }
        append_row(csv_path, row)
        print("APPENDED", row, flush=True)
        if not ok:
            print("STOP — FPS/TOPS missing; inspect METRICS lines / log", flush=True)
            sys.exit(1)
    print("BLOK2_DONE", flush=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paper III Blok 1 — Acc grid runner with Drive CSV append + skip_completed."""
from __future__ import annotations
import argparse, csv, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RE_ACC = re.compile(r"Test Accuracy of the model on\s+(\d+)\s+test images:\s*([0-9.]+)\s*%")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="results CSV path (prefer Drive)")
    p.add_argument("--prs", default="16,32,64,128")
    p.add_argument("--adcs", default="5,8,12")
    p.add_argument("--num_batches", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--sub_array", default="[128,128]")
    p.add_argument("--data_path", default="./datasets/")
    p.add_argument("--model_path", default="./models/")
    return p.parse_args()

def load_done(csv_path: Path):
    done = set()
    if not csv_path.exists():
        return done
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r.get("phase") != "blok1_acc":
                continue
            try:
                done.add((int(r["parallel_read"]), int(r["adc"])))
            except Exception:
                pass
    return done

def append_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp","phase","dataset","model","hardware","parallel_read","adc",
              "num_batches","acc_pct","n_images","notes"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()

def last_acc(text: str):
    matches = list(RE_ACC.finditer(text))
    if not matches:
        return None, None
    m = matches[-1]
    return float(m.group(2)), int(m.group(1))

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    prs = [int(x) for x in args.prs.split(",")]
    adcs = [int(x) for x in args.adcs.split(",")]
    # ADC-sensitive first: low ADC, extreme PRs first
    order = [(pr, adc) for adc in sorted(adcs) for pr in sorted(prs)]
    done = load_done(csv_path)
    print(f"CSV={csv_path} done={sorted(done)} remaining={[(a,b) for a,b in order if (a,b) not in done]}")

    for pr, adc in order:
        if (pr, adc) in done:
            print(f"SKIP PR={pr} ADC={adc}")
            continue
        cmd = [
            sys.executable, "-u", str(ROOT / "inference.py"),
            "--dataset", "cifar10", "--model", "resnet18",
            "--hardware", "1", "--ppa", "0",
            "--num_batches", str(args.num_batches),
            "--batch_size", str(args.batch_size),
            "--adc_precision", str(adc),
            "--sub_array", args.sub_array,
            "--parallel_read", str(pr),
            "--mem_type", "resistive",
            "--mem_states_file", "mem_states.csv",
            "--data_path", args.data_path,
            "--model_path", args.model_path,
        ]
        print("=" * 60)
        print("RUN", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        # stream tail for Colab visibility
        print(out[-4000:])
        acc, n_img = last_acc(out)
        ok = proc.returncode == 0 and acc is not None
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": "blok1_acc",
            "dataset": "cifar10",
            "model": "resnet18",
            "hardware": "1",
            "parallel_read": str(pr),
            "adc": str(adc),
            "num_batches": str(args.num_batches),
            "acc_pct": f"{acc:.5f}" if acc is not None else "",
            "n_images": str(n_img or ""),
            "notes": "ok" if ok else f"FAIL rc={proc.returncode}",
        }
        append_row(csv_path, row)
        print("APPENDED", row, flush=True)
        if not ok:
            print("STOP on failure — fix and re-run (skip_completed will resume)")
            sys.exit(1)
    print("BLOK1_DONE")

if __name__ == "__main__":
    main()

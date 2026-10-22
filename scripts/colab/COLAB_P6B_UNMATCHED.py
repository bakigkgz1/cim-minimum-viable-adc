# PAPER3 P6b — unmatched stock-LO HW demo (self-contained Colab cell).
#
# Provenance: Appendix A forbids claim numbers from archive blok* CSVs.
# This re-runs NeuroSIM PPA at stock Param.cpp levelOutput=128 (no 2^ADC patch)
# for PR=128 × ADC∈{5,6,8,12} inside full_rerun_v1/.
#
# One Param.cpp, one rebuild, one sha for all four rows.
# Accuracy path is not the claim: hardware=1, ppa=1, num_batches=1 (same as P6).
#
# Drive out: full_rerun_v1/P6b_unmatched_demo.csv
#            full_rerun_v1/logs/Param_unmatched_lo128.cpp
#            full_rerun_v1/logs/hw_unmatched_PR128_ADC*.log
#
# Resume-safe. Paste CSV back to chat when PRE-SUBMIT-style COMPLETE prints.

from __future__ import annotations

import csv
import hashlib
import os
import re
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
LOG = OUT / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

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
assert (models_dst / "resnet18_cifar10.pth").exists(), "missing resnet18_cifar10.pth on Drive"

pq = ROOT / "pytorch-quantization"
rc = os.system(
    f'{sys.executable} -c "from pytorch_quantization.cim.modules import macro"'
)
if rc != 0:
    print("Installing NeuroSim pytorch-quantization ...", flush=True)
    rc2 = os.system(f"{sys.executable} -m pip -q install -e {pq}")
    assert rc2 == 0


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def last_match(rx: re.Pattern, text: str) -> str:
    ms = list(rx.finditer(text))
    return ms[-1].group(1) if ms else ""


def write_csv(path: Path, rows: list[dict]):
    assert rows
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def force_stock_lo128() -> tuple[int, str]:
    """Stock Param.cpp costing: levelOutput=128 for every Python ADC."""
    param = ROOT / "NeuroSIM" / "Param.cpp"
    text = param.read_text()
    new, n = re.subn(
        r"levelOutput\s*=\s*\d+\s*;",
        "levelOutput = 128;",
        text,
        count=1,
    )
    assert n == 1, "levelOutput assign not found"
    param.write_text(new)
    snap = LOG / "Param_unmatched_lo128.cpp"
    snap.write_text(new)
    print("Param.cpp -> levelOutput=128", flush=True)
    print("make NeuroSIM ...", flush=True)
    rc = os.system(f"cd {ROOT / 'NeuroSIM'} && make -j2")
    assert rc == 0, "NeuroSIM make failed"
    ph = sha_text(new)
    print("param_sha", ph, "snap", snap, flush=True)
    return 128, ph


def run_hw_unmatched(pr: int, adc: int, lo: int, ph: str) -> dict:
    tag = "PR%s_ADC%s_unmatched" % (pr, adc)
    log = LOG / ("hw_unmatched_%s.log" % tag)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "inference.py"),
        "--dataset",
        "cifar10",
        "--model",
        "resnet18",
        "--hardware",
        "1",
        "--ppa",
        "1",
        "--num_batches",
        "1",
        "--batch_size",
        "64",
        "--parallel_read",
        str(pr),
        "--adc_precision",
        str(adc),
        "--sub_array",
        "[128,128]",
        "--data_path",
        "./datasets/",
        "--model_path",
        "./models/",
        "--test_name",
        "p6b_%s" % tag,
    ]
    t0 = time.time()
    print("HW unmatched", tag, "LO=%s sha=%s" % (lo, ph), flush=True)
    with log.open("w") as f:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True
        )
    text = log.read_text(errors="ignore")
    assert proc.returncode == 0, text[-4000:]
    re_eff = re.compile(
        r"Energy Efficiency TOPS/W \(Pipelined Process\):\s*([\d.eE+-]+)"
    )
    re_fps = re.compile(r"Throughput FPS \(Pipelined Process\):\s*([\d.eE+-]+)")
    re_tops = re.compile(r"Throughput TOPS \(Pipelined Process\):\s*([\d.eE+-]+)")
    re_area = re.compile(r"ChipArea\s*:\s*([\d.eE+-]+)")
    re_adc_e = re.compile(
        r"ADC \(or S/As and precharger for SRAM\) readDynamicEnergy is\s*:\s*([\d.eE+-]+)"
    )
    re_acc_e = re.compile(
        r"Accumulation Circuits.*?readDynamicEnergy is\s*:\s*([\d.eE+-]+)"
    )
    re_oth_e = re.compile(
        r"Other Peripheries.*?readDynamicEnergy is\s*:\s*([\d.eE+-]+)"
    )
    row = {
        "ts": utc_now(),
        "tag": tag,
        "pr": pr,
        "adc": adc,
        "levelOutput": lo,
        "param_sha": ph,
        "matched_lo": 0,
        "tops_w": last_match(re_eff, text),
        "fps": last_match(re_fps, text),
        "tops": last_match(re_tops, text),
        "area": last_match(re_area, text),
        "adc_E_pJ": last_match(re_adc_e, text),
        "accum_E_pJ": last_match(re_acc_e, text),
        "other_E_pJ": last_match(re_oth_e, text),
        "seconds": round(time.time() - t0, 1),
        "log": str(log),
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


path = OUT / "P6b_unmatched_demo.csv"
corners = [
    (128, 5),
    (128, 6),
    (128, 8),
    (128, 12),
]
rows = []
if path.exists():
    with path.open() as f:
        rows = list(csv.DictReader(f))
done = {(int(r["pr"]), int(r["adc"])) for r in rows}
print("P6b already", sorted(done), flush=True)

lo = ph = None
if len(done) < len(corners):
    lo, ph = force_stock_lo128()
    if rows:
        # Resume must keep the same Param sha; refuse mixed snapshots.
        old = {r.get("param_sha") for r in rows}
        if old - {ph, ""}:
            raise SystemExit(
                "P6b CSV sha %s != live Param sha %s. Delete P6b_unmatched_demo.csv and rerun."
                % (old, ph)
            )

for pr, adc in corners:
    if (pr, adc) in done:
        print("SKIP unmatched PR=%s ADC=%s" % (pr, adc), flush=True)
        continue
    rows.append(run_hw_unmatched(pr, adc, lo, ph))
    write_csv(path, rows)
    done.add((pr, adc))

print("=" * 70, flush=True)
print("P6b UNMATCHED COMPLETE", path, flush=True)
print("All rows share levelOutput=128 and one param_sha.", flush=True)
for r in rows:
    print(
        "  ADC=%s  TOPS/W=%s  area=%s  FPS=%s  sha=%s"
        % (r["adc"], r["tops_w"], r["area"], r["fps"], r["param_sha"]),
        flush=True,
    )
print("Paste back to chat:", path, flush=True)
print("=" * 70, flush=True)
